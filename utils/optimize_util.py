import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from .stencil_util import stencilOrder, stencil_6, stencil_26
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from .data_io import save_nifty_data
import matplotlib.pyplot as plt

def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")

    return torch.softmax(x_masked, **kwargs)

def parse_params(param_fname):
    import yaml
    assert param_fname.endswith('.yaml')
    with open(param_fname, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['param_fname'] = param_fname
    return params

def write_exp_log(log_fname, params):
    import git
    repo = git.Repo(path = os.path.dirname(os.path.abspath(__file__)))
    sha = repo.head.object.hexsha
    branch = repo.active_branch.name
    
    if not os.path.exists(os.path.dirname(log_fname)):
        os.makedirs(os.path.dirname(log_fname))

    with open (log_fname, 'w') as f:
        f.write(f'git branch: {branch}\n')
        f.write(f'git sha: {sha}\n')
        for k,v in params.items():
            f.write(f'{k}: {v}\n')

def make_results(rad, tan, peaks, max_peaks):
    '''
    create results dict from variables
    '''
    rad_one_hot = F.one_hot(rad.argmax(dim=-1), num_classes=max_peaks+1)
    tan_one_hot = F.one_hot(tan.argmax(dim=-1), num_classes=max_peaks+1)

    rad_peak, rad_mask = peak_sel(rad_one_hot, peaks)
    tan_peak, tan_mask = peak_sel(tan_one_hot, peaks)
    
    result = {'rad_peak':rad_peak, 'tan_peak':tan_peak, 'rad_mask':rad_mask, 'tan_mask':tan_mask}
    return result

def peak_sel(E, peaks, to_numpy=True):
    '''
    E: (X,Y,Z,max_peak+1)
    peaks: (X,Y,Z,max_peak,3)
    '''
    E_sel = E[..., :-1]
    sel_peak = (peaks*E_sel.unsqueeze(-1)).sum(dim=-2)
    sel_peak_norm = torch.norm(sel_peak, dim=-1)
    E_mask = sel_peak_norm > 0
    if to_numpy:
        if sel_peak.is_cuda:
            sel_peak = sel_peak.cpu()
            sel_peak = sel_peak.detach().numpy()
        if E_mask.is_cuda:
            E_mask = E_mask.cpu()
            E_mask = E_mask.detach().numpy()
    return sel_peak, E_mask

def fill_0_with_normal(peaks, normal, ctx_mask, scale=0.1):
    peaks_norm = np.linalg.norm(peaks, axis=-1)
    #fill peaks_norm ==0 with normal*peak_threshold
    mask = peaks_norm == 0
    #exclude voxels outside cortex
    normal[ctx_mask==0] = 0
    mask = np.logical_and(mask, ctx_mask[...,None])
    peaks += mask[...,None]*normal[...,None,:]*scale
    peaks[:,:,:,-1,:] = normal*scale

def gradient(x):
    '''
    x: (X,Y,Z)
    return (X,Y,Z,3)
    '''
    xx, yy, zz = torch.gradient(x)
    return torch.stack([xx, yy, zz], dim=-1)

def vector_angle(x, y, eps=1e-8, do_normalize=True):
    '''
    x: (X,Y,Z,...,3)
    y: (X,Y,Z,...,3)
    return (X,Y,Z) range (0,1)
    '''
    if do_normalize:
        x_l2_norm = torch.norm(x, dim=-1)+eps
        y_l2_norm = torch.norm(y, dim=-1)+eps
        x_normalized = x/x_l2_norm.unsqueeze(-1)
        y_normalized = y/y_l2_norm.unsqueeze(-1)
    else:
        x_normalized = x
        y_normalized = y
    return torch.abs(torch.sum(x_normalized*y_normalized, dim=-1))

def get_patch(tensor, patch_size=3):

    X, Y, Z = tensor.shape[:3]

    # Unfold to get 3x3x3 patches around each voxel
    padded_tensor = F.pad(tensor, (0, 0, 1, 1, 1, 1, 1, 1), mode='constant')
    unfolded = padded_tensor.unfold(0, patch_size, 1).unfold(1, patch_size, 1).unfold(2, patch_size, 1)  # Shape: (X, Y, Z, 3, 3, 3, 3)

    patches = unfolded.reshape(X, Y, Z, 27, -1)
    return patches

def get_patch_stencil(tensor, stencil_order:stencilOrder=stencil_6):
    '''
    tensor: (...,X,Y,Z,C)
    return (...,X,Y,Z,n_stencil,C)
    '''
    assert tensor.dim() >= 4
    # x, y, z, c = tensor.shape
    x, y, z = tensor.shape[:3]
    tensor_list = []
    order_array = stencil_order.order
    px, nx = order_array[:,0].max(), -order_array[:,0].min()
    py, ny = order_array[:,1].max(), -order_array[:,1].min()
    pz, nz = order_array[:,2].max(), -order_array[:,2].min()

    padded_tensor = F.pad(tensor, (0, 0, nz, pz, ny, py, nx, px), mode='constant')
    for dx,dy,dz in order_array:
        tensor_list.append(padded_tensor[dx+nx:x+dx+nx, dy+ny:y+dy+ny, dz+nz:z+dz+nz, :])
    
    patches = torch.stack(tensor_list, dim=-2)
    return patches

def rad_label(peaks_label:np.ndarray, _rad:np.ndarray):
    '''
    peaks_label: (X,Y,Z,max_peak)
    _rad: (X,Y,Z,max_peak)
    '''
    _rad_index = _rad.argmax(axis=-1)
    rad_label_ = np.take_along_axis(peaks_label, _rad_index[..., np.newaxis], axis=-1).squeeze(axis=-1)
    return rad_label_

def initialize_variable(n_max_peak, device, peaks, peak_mask, normals, init_type):
    variable_shape = peaks.shape[:3] + (n_max_peak+1,)
    if init_type == 'random':
        rad = torch.randn(variable_shape, requires_grad=True, device=device)
        tan = torch.randn(variable_shape, requires_grad=True, device=device)
        
    elif init_type == 'normal_dis':
        #calulate the angle difference between peaks and normal
        angles = vector_angle(peaks, normals.unsqueeze(-2), do_normalize=True)

        rad = torch.zeros(variable_shape, dtype=torch.float32, device=device, requires_grad=False)
        rad[..., :-1] = angles - 0.707
        rad[peak_mask == 0] = -float("inf")

        tan = torch.zeros(variable_shape, dtype=torch.float32, device=device, requires_grad=False)
        tan[..., :-1] = -angles + 0.707
        tan[peak_mask == 0] = -float("inf")

        rad.requires_grad_()
        tan.requires_grad_()

    else:
        raise ValueError('init_type not supported')

    return rad, tan

def save_results_nifty(results, ref_fname, out_prefix):
    '''
    results: dict
    '''
    rad_peak_dir_name = out_prefix + '_rad_peaks.nii.gz'
    save_nifty_data(rad_peak_dir_name, results['rad_peak'], ref_file=ref_fname)

    tan_peak_dir_name = out_prefix + '_tan_peaks.nii.gz'
    save_nifty_data(tan_peak_dir_name, results['tan_peak'], ref_file=ref_fname)

    rad_zero_name = out_prefix + '_rad_mask.nii.gz'
    save_nifty_data(rad_zero_name, results['rad_mask'].astype(np.uint8), ref_file=ref_fname)

    tan_zero_name = out_prefix + '_tan_mask.nii.gz'
    save_nifty_data(tan_zero_name, results['tan_mask'].astype(np.uint8), ref_file=ref_fname)

def save_ckpt(epoch, rad, tan, peaks, max_peaks, ckpt_params):
    if not os.path.exists(ckpt_params['ckpt_dir']):
        os.makedirs(ckpt_params['ckpt_dir'])

    ckpt_prefix = os.path.join(ckpt_params['ckpt_dir'], f'ckpt_{epoch}')
    results = make_results(rad, tan, peaks, max_peaks)
    save_results_nifty(results, ckpt_params['ref_fpath'], ckpt_prefix)
    
def optimize_one(peaks:np.ndarray, normals:np.ndarray, params:dict, neighbor_mask=None, stencil_order=stencil_6):
    '''
    we don't need mask anymore cause peaks contains the mask information
    normals: (X,Y,Z,3)
    params: parameters from yaml
    peaks: (X,Y,Z,max_peak,3)
    init_sdf: (X,Y,Z)
    neighbor_mask: (X,Y,Z,n_stencil)
    '''
    if params['device'] == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device(params['device'] if torch.cuda.is_available() else "cpu")

    if neighbor_mask is None:
        neighbor_mask = np.ones(normals.shape[:3] + (len(stencil_order),), dtype=np.float32)

    peaks = torch.tensor(peaks, device=device, requires_grad=False)
    normals = torch.tensor(normals, device=device, requires_grad=False)
    neighbor_mask = torch.tensor(neighbor_mask, device=device, requires_grad=False)
    # sdf = torch.tensor(init_sdf, device=device, requires_grad=True)

    #create peak mask which has positive peaks
    peak_l2_norm = torch.norm(peaks, dim=-1)
    peak_mask = peak_l2_norm > 0
    #pad last dimension with 1
    peak_mask = F.pad(peak_mask, (0,1), value=True)

    #initialize variables 
    rad, tan = initialize_variable(params['max_peaks'], device, peaks, peak_mask, normals, params['init_type'])

    log_tag = ''
    log_tag = log_tag + params['log_tag'] if 'log_tag' in params.keys() else log_tag
    log_tag = log_tag + params['subj'] if 'subj' in params.keys() else log_tag
    log_tag = log_tag + params['hemi'] if 'hemi' in params.keys() else log_tag
    log_fpath = 'runs/' + log_tag + datetime.now().strftime("%Y%m%d-%H%M%S")

    # if 'log_tag' in params.keys():
    #     log_fpath = 'runs/' + params['log_tag'] + datetime.now().strftime("%Y%m%d-%H%M%S")
    # else:
    #     log_fpath = 'runs/'+ datetime.now().strftime("%Y%m%d-%H%M%S")

    tb_writer = logger(log_fpath)
    img_params = params['img_log_params']
    tb_writer.init_plot(img_params['img_roi'], img_params['n_row'], img_params['n_col'])
    loss_fn = OptimizeModel(params['max_peaks'], peaks, peak_mask, normals, neighbor_map=neighbor_mask, stencil_order=stencil_order, writer=tb_writer, **params['loss_params'])
    optimizer = optim.Adam([rad, tan], lr=params['lr'])
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(params['num_epoch']):  # Number of epochs
        tb_writer.set_epoch(epoch)
        optimizer.zero_grad()  # Zero the gradients

        loss = loss_fn(rad, tan)

        loss.backward(retain_graph=True)  # Backpropagate the gradients

        optimizer.step()  # Update the variables (rad)
        # scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        #save ckpt for debug
        ckpt_freq = params['ckpt_params']['ckpt_freq']
        if ckpt_freq != 0 and epoch % ckpt_freq == 0:
            save_ckpt(epoch, rad, tan, peaks, params['max_peaks'], params['ckpt_params'])

    tb_writer.close()

    return make_results(rad, tan, peaks, max_peaks=params['max_peaks'])

class logger(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super(logger, self).__init__(*args, **kwargs)
        self.epoch = 0
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def init_plot(self, roi, n_row, n_col):
        '''
        roi:[x0,x1,y0,y1,z0,z1]
        '''
        self.n_row = n_row
        self.n_col = n_col
        self.roi = roi
        assert len(roi) == 6
        assert n_row > 0 and n_col > 0
        n_slices =  roi[5] - roi[4]  # Number of slices to visualize
        assert n_slices == self.n_row * self.n_col, "Mismatch in grid size and slices"


    def add_scalar(self, tag, scalar_value, walltime=None, new_style=False, double_precision=False):
        # return super().add_scalar(tag, scalar_value, self.epoch, walltime, new_style, double_precision)
        pass
    
    def add_peak_image(self, tag, img, color=None):
        '''
        img: (X, Y, Z, 3) tensor representing the vector field
        color: (R, G, B) tuple for uniform vector coloring, or None for direction-based coloring
        '''
        x0, x1, y0, y1, z0, z1 = self.roi

        fig, axes = plt.subplots(self.n_row, self.n_col, figsize=(15, 15))
        axes = axes.flatten()  # Flatten for easy iteration

        img_tensor_np = img[x0:x1, y0:y1, z0:z1].detach().cpu().numpy()
        
        for i, z in enumerate(range(z0, z1)):
            ax = axes[i]
            
            # Extract the slice from the vector field
            slice_vectors = img_tensor_np[..., z - z0,:]  # Shape: (X, Y, 3)
            
            # Compute positions
            X, Y = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1))
            U, V, W = slice_vectors[..., 0], slice_vectors[..., 1], slice_vectors[..., 2]  # Only X-Y plane components
            
            # Determine colors
            if color is None:
                # Normalize vector direction for coloring
                # Compute the overall magnitude at each point (avoid division by zero).
                mag = np.sqrt(U**2 + V**2 + W**2)
                mag[mag == 0] = 1

                # Compute colors:
                # Use the absolute values of each component so that an arrow purely in the x, y, or z direction
                # will have a color of [1, 0, 0] (red), [0, 1, 0] (green), or [0, 0, 1] (blue), respectively.
                R = np.abs(U) / mag
                G = np.abs(V) / mag
                B = np.abs(W) / mag
                colors = np.stack((R, G, B), axis=-1)
            else:
                colors = np.full_like(slice_vectors, fill_value=color)  # Uniform color
            colors = colors.reshape(-1, 3)
            # Plot quiver (vector field)
            ax.quiver(X, Y, U*10, V*10, color=colors[..., :3], scale=50, width=0.002)
            ax.set_title(f"Slice Z={z}")
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        # Convert the matplotlib figure to an image
        plt.savefig(f'tmp/{tag}_{self.epoch}.png')
        plt.close(fig)
        # buf = BytesIO()
        # plt.savefig(buf, format='png')
        # plt.close(fig)  # Close to prevent memory issues
        # buf.seek(0)
        
        # image = Image.open(buf)
        # image = image.convert("RGB")  # Ensure RGB mode
        # image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # Convert to (C, H, W) for TensorBoard

        # self.add_image(tag, image_tensor, self.epoch)


    def add_grey_img(self, tag, img):
        '''
        img: (X, Y, Z) tensor representing the scalar field
        '''
        x0, x1, y0, y1, z0, z1 = self.roi

        fig, axes = plt.subplots(self.n_row, self.n_col, figsize=(15, 15))
        axes = axes.flatten()  # Flatten for easy iteration

        img_tensor_np = img[x0:x1, y0:y1, z0:z1].detach().cpu().numpy()
        for i, z in enumerate(range(z0, z1)):
            ax = axes[i]
            
            # Extract the slice from the vector field
            slice_tensor = img_tensor_np[..., z - z0]  # Shape: (X, Y)
            ax.imshow(slice_tensor, cmap='gray')
            ax.set_title(f"Slice Z={z}, max={slice_tensor.max()}, min={slice_tensor.min()}")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(f'tmp/{tag}_{self.epoch}.png')
        plt.close(fig)


    def add_sel_img(self,tag,img,z_slice):
        '''
        img: (X, Y, Z, 6) 
        '''
        x0, x1, y0, y1, z0, z1 = self.roi

        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.flatten()  # Flatten for easy iteration

        img_tensor_np = img[x0:x1, y0:y1, z_slice,:].detach().cpu().numpy()
        for i, z in enumerate(range(0,16)):
            ax = axes[i]
            
            # Extract the slice from the vector field
            slice_tensor = img_tensor_np[:,z,:]  # Shape: (X, 6)
            ax.imshow(slice_tensor, cmap='gray')
            ax.set_title(f"Slice Z={z}, max={slice_tensor.max()}, min={slice_tensor.min()}")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(f'tmp/{tag}_{self.epoch}.png')
        plt.close(fig)

    def add_sel_dbg_img(self, tag, sel, mask, ent, z_slice):
        '''
        sel: (X, Y, Z, 6)
        mask: (X, Y, Z)
        ent: (X, Y, Z)
        #TODO set entropy to green if voxel is out of ctx_mask
        '''
        
        x0, x1, y0, y1, z0, z1 = self.roi
        roi_shape = (x1-x0, y1-y0,1)
        sel_tensor_np = sel[x0:x1, y0:y1, z_slice,:].detach().cpu().numpy()
        mask_tensor_np = mask[x0:x1, y0:y1, z_slice].detach().cpu().numpy()
        ent_tensor_np = ent[x0:x1, y0:y1, z_slice].detach().cpu().numpy()

        ent_color = ent_tensor_np[...,None]/2*np.array([1,0,0])
        sel_tensor_color = sel_tensor_np[...,None]*np.array([1,1,1])
        sel_tensor_color[mask_tensor_np==0] = [1,1,0]
        img_tensor_np = np.concatenate([sel_tensor_color, ent_color[...,None,:]], axis=-2)#（16，16，7，3）
        
        img = np.zeros((32,56,3))
        # img[17, ...] = [1,1,0]
        for i_row in range(2):
            for i_col in range(8):
                img[roi_shape[0]*i_row:roi_shape[0]*(i_row+1), 7*i_col:7*(i_col+1), :] = img_tensor_np[:, i_row*8+i_col,:,:]


        # Display the image
        # plt.axis("off")  # Hide axis
        # plt.savefig(f'tmp/{tag}_{self.epoch}.png')
        # plt.close()
                
        image_tensor = torch.tensor(img).permute(2, 0, 1)  # Convert (H, W, 3) -> (3, H, W)
        #upsample by 4x
        image_tensor = F.interpolate(image_tensor.unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0)

        # Add image to TensorBoard
        return super().add_image(tag, image_tensor, global_step=self.epoch)

class JSDiv(nn.Module):
    
    def __init__(self, eps=1e-8):
        super(JSDiv, self).__init__()
        self.eps = eps
    
    def forward(self, net_1_probs, net_2_probs):

        m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(torch.log(net_1_probs+self.eps), m, reduction="sum") 
        loss += F.kl_div(torch.log(net_2_probs+self.eps), m, reduction="sum") 
     
        return (0.5 * loss)
    
class unitaryLoss(nn.Module):
    def __init__(self, N, lambda1, lambda2, lambda3, writer:logger=None):

        super(unitaryLoss, self).__init__()
        self.N = N
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.writer = writer


    def forward(self, rad_peaks, tan_peaks, rad_mask, tan_mask):
        '''
        peaks: (X,Y,Z,3)
        return (X,Y,Z)
        '''

        rad_loss = vector_angle(rad_peaks, self.N)*rad_mask
        tan_loss = vector_angle(tan_peaks, self.N)*tan_mask
        # amp_reg = (torch.norm(rad_peaks, dim=-1)*rad_mask + torch.norm(tan_peaks, dim=-1)*tan_mask)
        # amp_reg = -self.lambda3*(torch.norm(rad_peaks, dim=-1)*rad_mask).sum()/rad_mask.sum() \
        #     - self.lambda3*(torch.norm(tan_peaks, dim=-1)*tan_mask).sum()/tan_mask.sum()
        amp_reg = 0

        rad_loss = -self.lambda1*rad_loss.sum()/rad_mask.sum()
        tan_loss = self.lambda2*tan_loss.sum()/tan_mask.sum()

        self.writer.add_scalar('U_rad_loss', rad_loss.item())
        self.writer.add_scalar('U_tan_loss', tan_loss.item())
        # self.writer.add_scalar('U_amp_reg', amp_reg.item())
        return rad_loss + tan_loss + amp_reg

class pairwiseLoss(nn.Module):
    def __init__(self, lambda4, lambda5, lambda6, neighbor_map, stencil_order:stencilOrder, writer:logger=None, eps=1e-8) -> None:
        '''
        neighbor_map: (X,Y,Z, n_stencil)
        '''
        super().__init__()
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.lambda6 = lambda6
        self.neighbor_map = neighbor_map
        assert len(stencil_order) == neighbor_map.shape[-1]
        self.stencil_order = stencil_order
        self.writer = writer
        self.eps = eps

    def forward(self, rad_peak, tan_peak, rad_mask, tan_mask):
        '''
        rad_peak: (X,Y,Z,3)
        tan_peak: (X,Y,Z,3)
        rad_mask: (X,Y,Z) 1: non-zero, 0: zero
        tan_mask: (X,Y,Z) 1: non-zero, 0: zero
        '''
        rad_peaks_neighbors = get_patch_stencil(rad_peak, self.stencil_order)

        #(X,Y,Z, n_stencil)
        rad_mask_neighbors = get_patch_stencil(rad_mask.unsqueeze(-1), self.stencil_order)
        # tan_mask_neighbors = get_patch_stencil(tan_mask.unsqueeze(-1), self.stencil_order)

        #avoid averaging the angles of 0s, but still consider the length
        rad_angles = -vector_angle(rad_peak.unsqueeze(-2), rad_peaks_neighbors, do_normalize=True)
        #(X,Y,Z, n_stencil)
        rad_lengths = (torch.norm(rad_peak.unsqueeze(-2), dim=-1) - torch.norm(rad_peaks_neighbors, dim=-1))**2

        tan_peaks_neighbors = get_patch_stencil(tan_peak, self.stencil_order)
        tan_lengths = (torch.norm(tan_peak.unsqueeze(-2), dim=-1) - torch.norm(tan_peaks_neighbors, dim=-1))**2
        assert rad_angles.shape[-1] == self.neighbor_map.shape[-1], 'ensure the number of stencils are the same'
        rad_angle_mask = rad_mask_neighbors.squeeze(-1)*self.neighbor_map*rad_mask.unsqueeze(-1) #non-zero peak && non-zero neighbor && neighbor relation exists
        
        rad_ang_loss = self.lambda4*(rad_angles * rad_angle_mask).sum()/rad_angle_mask.sum()

        rad_proj_tan = (rad_peak*tan_peak).sum(dim=-1)/(torch.norm(rad_peak, dim=-1) + self.eps)/(torch.norm(tan_peak, dim=-1) + self.eps)
        rad_proj_tan = rad_proj_tan.unsqueeze(-1)*tan_peak

        tan_plane_normal = rad_peak - rad_proj_tan
        tan_peaks_neighbors = get_patch_stencil(tan_plane_normal, self.stencil_order)

        tan_angles = -vector_angle(tan_plane_normal.unsqueeze(-2), tan_peaks_neighbors, do_normalize=True)

        rad_lengths_loss = self.lambda6 * (rad_lengths * self.neighbor_map).sum()/ self.neighbor_map.sum()
        tan_lengths_loss = self.lambda6 * (tan_lengths * self.neighbor_map).sum()/ self.neighbor_map.sum()
        lengths_loss = rad_lengths_loss + tan_lengths_loss

        #only average the angles of non-zero rad_peaks, doesnt matter if the tan_peaks are zero
        #TODO consider the case when rad_peak is zero but tan_peak is not
        #SOL JS div regularization will eliminates the rad_peak selection
        tan_loss = self.lambda5*(tan_angles * rad_angle_mask).sum()/rad_angle_mask.sum()

        #write to tensorboard
        self.writer.add_scalar('P_rad_ang_loss', rad_ang_loss.item())
        self.writer.add_scalar('P_tan_loss', tan_loss.item())
        self.writer.add_scalar('P_lengths_loss', lengths_loss.item())

        return rad_ang_loss + tan_loss + lengths_loss
    
class OptimizeModel(nn.Module):
    def __init__(self, max_peaks, peaks, peak_mask, normal, neighbor_map,
                lambda1, lambda2, lambda3, lambda4, lambda5, lambda6,
                entropy_param, js_param,peak_choose_type,
                gumbel_tau=0.1, eps=1e-8, entropy_threshold=0,
                stencil_order=stencil_6, writer:logger=None):
        '''
        minimize this loss
        peaks: (X,Y,Z,max_peak,3)
        normal: (X,Y,Z,3)
        '''
        super(OptimizeModel, self).__init__()
        self.normal = normal
        self.stencil_order = stencil_order
        self.writer = writer
        self.neighbor_map = neighbor_map
        self.eps = eps
        # entropy_reg which reduce the entropy of the peak selection
        self.entropy_reg = entropyReg(entropy_param, js_param, eps, writer=self.writer)

        self.peak_selector = peakSelector(peak_choose_type, peak_mask, gumbel_tau, peaks, max_peaks, entropy_threshold, eps, writer=self.writer)
        self.ULoss = unitaryLoss(self.normal, lambda1, lambda2, lambda3, writer=self.writer)
        self.PLoss = pairwiseLoss(lambda4=lambda4, lambda5=lambda5, lambda6=lambda6, neighbor_map=neighbor_map, stencil_order=self.stencil_order, writer=self.writer)

        self.peak_mask = peak_mask
        self.entropy_threshold = entropy_threshold
    
    def forward(self, rad, tan):
        '''
        P_rad: (X,Y,Z,max_peaks+1) probability of choosing each peak and being zero
        P_tan: (X,Y,Z,max_peaks+1) probability of choosing each peak and being zero
        #last digit is the probability of being zero
        '''
        P_rad, P_tan, rad_peak, tan_peak = self.peak_selector(rad, tan)

        
        rad_entropy = -(P_rad * torch.log(P_rad + self.eps)).sum(dim=-1)  # Shape (x, y, z)
        tan_entropy = -(P_tan * torch.log(P_tan + self.eps)).sum(dim=-1)  # Shape (x, y, z)

        if self.entropy_threshold > 0:
            #detach rad_peak, tan_peak for low entropy voxels
            rad_entropy_mask = rad_entropy < self.entropy_threshold
            tan_entropy_mask = tan_entropy < self.entropy_threshold

            fixed_rad_peak = rad_peak.clone()
            fixed_rad_peak[rad_entropy_mask] = fixed_rad_peak[rad_entropy_mask].detach()

            fixed_tan_peak = tan_peak.clone()
            fixed_tan_peak[tan_entropy_mask] = fixed_tan_peak[tan_entropy_mask].detach()

            rad_peak = fixed_rad_peak
            tan_peak = fixed_tan_peak


        # self.writer.add_sel_dbg_img('rad_sel_dbg', P_rad[..., -1], self.peak_mask, rad_entropy, z_slice=60)
        # self.writer.add_sel_dbg_img('tan_sel_dbg', P_tan[..., -1], self.peak_mask, tan_entropy, z_slice=60)

        #TODO, compute non_zero_mask differentially from P_rad, P_tan
        rad_peak_norm = torch.norm(rad_peak, dim=-1)
        rad_peak_non_zero_mask = rad_peak_norm/ (rad_peak_norm + self.eps)

        tan_peak_norm = torch.norm(tan_peak, dim=-1)
        tan_peak_non_zero_mask = tan_peak_norm/ (tan_peak_norm + self.eps)


        # rad_weight = (torch.exp(rad_entropy)*rad_peak_non_zero_mask).detach()
        # tan_weight = (torch.exp(tan_entropy)*tan_peak_non_zero_mask).detach()
        
        # print(f"rad mask sum: {rad_weight.sum().item()}, tan mask sum: {tan_weight.sum().item()}")
        ULoss = self.ULoss(rad_peak, tan_peak, rad_peak_non_zero_mask, tan_peak_non_zero_mask)
        PLoss = self.PLoss(rad_peak, tan_peak, rad_peak_non_zero_mask, tan_peak_non_zero_mask)

        reg = self.entropy_reg(rad_entropy, tan_entropy, P_rad, P_tan)

        total_loss = ULoss + PLoss + reg
        self.writer.add_scalar('total_loss', total_loss.item())
        return total_loss
    
class entropyReg(nn.Module):
    def __init__(self, entropy_param, js_param, eps, writer:logger=None):
        super(entropyReg, self).__init__()
        self.eps = eps
        self.js_div = JSDiv(self.eps)
        self.writer = writer
        self.entropy_param = entropy_param
        self.js_param = js_param

    def forward(self, rad_entropy, tan_entropy, P_rad, P_tan):
        entropy_reg = self.entropy_param*(rad_entropy.sum() + tan_entropy.sum())
        self.writer.add_scalar('entropy_reg', entropy_reg.item())

        #sel_reg which avoid repeated selection of the same peak, increase js divergence of P_rad and P_tan
        #TODO how to make it a hard constraint?
        P_rad_peak = P_rad[...,:-1]/(P_rad[...,:-1].sum(dim=-1, keepdim=True) + self.eps)
        P_tan_peak = P_tan[...,:-1]/(P_tan[...,:-1].sum(dim=-1, keepdim=True) + self.eps)
        js_reg = -self.js_param*self.js_div(P_rad_peak, P_tan_peak)
        self.writer.add_scalar('js_reg', js_reg.item())
        return js_reg + entropy_reg
        
class peakSelector(nn.Module):
    def __init__(self, peak_choose_type, peak_mask,
                gumbel_tau, peaks, n_max_peaks,
                entropy_threshold, eps,
                writer:logger) -> None:
        super().__init__()
        assert peak_choose_type in ['soft', 'gumbel', 'gumbel_hard', 'hard']
        self.peak_choose_type = peak_choose_type
        self.gumbel_hard = True if peak_choose_type == 'gumbel_hard' else False
        self.gumbel_tau = gumbel_tau
        self.peak_mask = peak_mask
        self.writer = writer
        self.eps = eps
        self.entropy_threshold = entropy_threshold

        self.max_peaks = n_max_peaks
        #pad last dimension with 0
        self.peaks = peaks
        assert self.peaks.shape[-2] == n_max_peaks
        self.peaks_padded = F.pad(peaks, (0,0,0,1), value=0)
        assert self.peaks_padded.shape[-2] == n_max_peaks + 1
        assert self.peaks_padded.shape[-1] == 3

    def forward(self, L_rad, L_tan):
        '''
        L_rad: (X,Y,Z,max_peak)
        L_tan: (X,Y,Z,max_peak)
        '''
        P_rad = masked_softmax(L_rad, self.peak_mask, dim=-1)
        P_tan = masked_softmax(L_tan, self.peak_mask, dim=-1)

        if self.peak_choose_type == 'soft':
            P_rad_expanded = P_rad.unsqueeze(-1)
            P_tan_expanded = P_tan.unsqueeze(-1)
        elif self.peak_choose_type == 'hard':
            #notice this is none differentiable
            P_rad_expanded = F.one_hot(P_rad.argmax(dim=-1), num_classes=self.max_peaks+1).unsqueeze(-1)
            P_tan_expanded = F.one_hot(P_tan.argmax(dim=-1), num_classes=self.max_peaks+1).unsqueeze(-1)
        elif 'gumbel' in self.peak_choose_type:
            #use gumbel softmax, if gumbel_hard, guarantee only one peak is chosen
            P_rad_expanded = F.gumbel_softmax(P_rad, tau=self.gumbel_tau, hard=self.gumbel_hard).unsqueeze(-1)
            P_tan_expanded = F.gumbel_softmax(P_tan, tau=self.gumbel_tau, hard=self.gumbel_hard).unsqueeze(-1)
        else:
            raise ValueError('peak_choose_type not supported')
        # self.writer.add_sel_img('rad_sel',P_rad_expanded[..., -1], z_slice=60)
        # self.writer.add_sel_img('tan_sel',P_tan_expanded[..., -1], z_slice=60)


        #after mul and sum, the rad_peak and tan_peak is either a peak or 0
        rad_peak = (P_rad_expanded*self.peaks_padded).sum(dim=-2)
        tan_peak = (P_tan_expanded*self.peaks_padded).sum(dim=-2)
        # self.writer.add_peak_image('rad_peak', rad_peak)
        # self.writer.add_peak_image('tan_peak', tan_peak)


        return P_rad, P_tan, rad_peak, tan_peak
    