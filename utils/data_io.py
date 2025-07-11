import os
import nibabel as nib
import numpy as np
import scipy.io
from scipy.io import loadmat
from .basic_util import scan

def parse_params(param_fname):
    import yaml
    assert param_fname.endswith('.yaml')
    with open(param_fname, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['param_fname'] = param_fname
    return params

def load_TissueMap(folder, scan:scan, compartment='intra-axonal', return_path=False):
    '''
    compartment: 0: intra-axonal fraction,
                 1: free water fraction,
                 2: isotropic low diffusivity fraction,
                 3: estimated low diffusivity
    '''
    compartment_dict = {'intra-axonal': 0, 'free_water': 1, 'iso_low_diff': 2, 'low_diff': 3}
    assert compartment in compartment_dict.keys()

    fpath = os.path.join(folder, scan.rel_path, 'Diffusion', f'TissueMap.nii.gz')
    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()
    return data[..., compartment_dict[compartment]]

def load_FS_thickness(folder, scan:scan, hemi, return_path=False):
    fpath = os.path.join(folder, scan.rel_path, 'surf', f'{hemi}.thickness')
    if return_path:
        return fpath
    data = nib.freesurfer.io.read_morph_data(fpath)
    return data

def load_NODDI_data(folder, scan:scan, hemi, modality='NDI', return_path=False):
    fpath = os.path.join(folder, scan.rel_path, 'NODDI', f'{hemi}_{modality}.nii.gz')
    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()
    return data

def load_DTI_seg(folder, scan:scan, return_path=False):
    # fpath = os.path.join(folder, subj, 'DTISpace', f'wmparc_DTISpace.nii.gz')
    fpath = os.path.join(folder, scan.rel_path, 'DTISpace', f'aparc+aseg_DTISpace.nii.gz')

    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()  
    return data

def load_DTI_surf_mask(folder, scan:scan, hemi, return_path=False):
    fpath = os.path.join(folder, scan.rel_path, 'DTISpace', f'{hemi}_surf_mask.nii.gz')
    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()
    return data

def load_DTI_ctx_mask(folder, scan:scan, hemi, return_path=False):
    fpath = os.path.join(folder, scan.rel_path, 'DTISpace', f'{hemi}_ctx_mask.nii.gz')
    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()
    return data

def load_DTI_surf_normal(folder, scan:scan, hemi, return_path=False):
    fpath = os.path.join(folder, scan.rel_path, 'DTISpace', f'{hemi}_surf_normal.nii.gz')
    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()
    return data

def load_DTI_ctx_normal(folder, scan:scan, hemi, return_path=False):
    fpath = os.path.join(folder, scan.rel_path, 'DTISpace', f'{hemi}_ctx_normal.nii.gz')
    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()
    return data

def load_surf_data(folder, scan:scan,  hemi, modality='FA', return_path=False):
    assert hemi in ['lh', 'rh']
    fpath = os.path.join(folder, scan.rel_path,  'Surf', f'{hemi}_white_{modality}.raw')
    if return_path:
        return fpath
    data = np.fromfile(fpath, dtype=np.float32)
    return data

def load_suvr_surf_data(folder, scan:scan,  hemi, return_path=False):
    assert hemi in ['lh', 'rh']
    fpath = os.path.join(folder, scan.rel_path,  'pet_uniform','AV1451_acpc_dc_restore_1mm_smoothed', f'{hemi}.pvc.subj.suvr.raw')
    if return_path:
        return fpath
    data = np.fromfile(fpath, dtype=np.float32)
    return data

def load_DTI_data_vol(folder, scan:scan,  modality='FA', return_path=False):
    fpath = os.path.join(folder, scan.rel_path, 'Diffusion', f'{modality}.nii.gz')
    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()
    return data

def load_annotation(folder, scan:scan, hemi, return_path=False, a2009s=False):
    assert hemi in ['lh', 'rh']
    if a2009s:
        fpath = os.path.join(folder, scan.rel_path,  'label', f'{hemi}.aparc.a2009s.annot')
    else:
        fpath = os.path.join(folder, scan.rel_path,  'label', f'{hemi}.aparc.annot')
    if return_path:
        return fpath
    labels,_,_ = nib.freesurfer.io.read_annot(fpath)
    return labels

def load_FOD_surf_component(folder, scan:scan,  hemi, component, load_raw=False, return_path=False):
    assert hemi in ['lh', 'rh']
    assert component in ['tang3D', 'vert3D']
    if load_raw:
        fpath = os.path.join(folder, scan.rel_path,  'FODProj', f'{hemi}_{component[0:3]}.raw')
        data = np.fromfile(fpath, dtype=np.float32)
    else:
        fpath = os.path.join(folder, scan.rel_path,  'DTISpace', f'{hemi}_tan_ver_proj.mat')
        data = loadmat(fpath)[component]
    if return_path:
        return fpath
    return data

def load_FOD_surf_component_vol(folder, scan:scan,  hemi, component, return_path=False):
    assert hemi in ['lh', 'rh']
    assert component in ['tang3D', 'vert3D']
    fpath = os.path.join(folder, scan.rel_path,  'FODProj', f'{hemi}_FOD_{component[0:3]}.nii.gz')
    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()
    return data

def load_FOD_ctx_component_vol(folder, scan:scan,  hemi, component, return_path=False):
    assert hemi in ['lh', 'rh']
    assert component in ['tang3D', 'vert3D']
    fpath = os.path.join(folder, scan.rel_path,  'FODProj', f'{hemi}_ctx_FOD_{component[0:3]}.nii.gz')
    if return_path:
        return fpath
    data = nib.load(fpath).get_fdata()
    return data

def load_wm_mesh(folder, scan:scan,  hemi, return_path=False, load_rm = False):
    '''
    Load surface mesh from freesurfer folder
    output: header(H), coord(V), normals(N), triangles(F)
    '''
    assert hemi in ['lh', 'rh']
    if load_rm:
        fpath = os.path.join(folder, scan.rel_path,  'DTISpace', f'{hemi}_white_DTISpace_rm.obj')
    else:
        fpath = os.path.join(folder, scan.rel_path,  'DTISpace', f'{hemi}_white_DTISpace.obj')
    if return_path:
        return fpath
    header_info, coord, normals, triangles = read_mni_obj(fpath)
    return header_info, coord, normals, triangles

def load_pial_mesh(folder, scan:scan,  hemi, return_path=False):
    '''
    Load surface mesh from freesurfer folder
    output: header(H), coord(V), normals(N), triangles(F)
    '''
    assert hemi in ['lh', 'rh']
    fpath = os.path.join(folder, scan.rel_path,  'DTISpace', f'{hemi}_pial_DTISpace.obj')
    if return_path:
        return fpath
    header_info, coord, normals, triangles = read_mni_obj(fpath)
    return header_info, coord, normals, triangles

def load_mesh(folder, scan:scan,  hemi, modality='sphere.reg' , return_path=False):
    '''
    Load surface mesh from freesurfer folder
    output: header(H), coord(V), normals(N), triangles(F)
    '''
    assert hemi in ['lh', 'rh']
    fpath = os.path.join(folder, scan.rel_path,  'surf', f'{hemi}.{modality}.obj')
    if return_path:
        return fpath
    header_info, coord, normals, triangles = read_mni_obj(fpath)
    return header_info, coord, normals, triangles

def load_mesh_attribute(folder, scan:scan,  hemi, modality='rad_temp', return_path=False):
    '''
    Load surface mesh from freesurfer folder
    output: numpy.ndarray
    '''
    assert hemi in ['lh', 'rh']
    fpath = os.path.join(folder, scan.rel_path, 'surf', f'{hemi}_{modality}')
    if return_path:
        return fpath
    else:
        return np.fromfile(fpath, np.float32)
    
def save_mesh_attribute(fname, data):
    with open(fname, 'wb') as f:
        data.astype(np.float32).tofile(f)

def save_surf_mesh(fname, H, V, N, F):
    new_H = list(H)
    new_H[1] = V.shape[0]
    new_H[2] = F.shape[0]
    new_H[5] = range(3, F.shape[0]*3+1,3)
    save_mni_obj(fname, new_H,V,N,F)

def save_nifty_data(fname, data:np.ndarray, ref_file=None, affine=None):
    '''
    Save numpy array to nifty file, either with reference file or affine matrix
    fname: str, file path
    data: np.ndarray, 3D array
    ref_file: str, reference file path
    affine: np.ndarray, 4x4 array
    '''
    assert ref_file is not None or affine is not None
    dir = os.path.dirname(fname)
    if not os.path.exists(dir):
        os.mkdir(dir)
    if ref_file is not None:
        affine = nib.load(ref_file).affine
    img2 = nib.Nifti1Image(data, affine)
    nib.save(img2, fname)

def save_subj_txt(txt_fname, subjs):
    with open(txt_fname, 'w') as f:
        for subj in subjs[:-1]:
            f.write(subj+'\n')
        f.write(subjs[-1])

"""
from yihao's code
"""
def list2str(array, two_deci=True, splitter=' '):
    """
    Join a list with spaces between elements.
    """
#     return ' '.join(str(a) for a in array)
    if two_deci:
        str_list = [f'{a:.2f}' for a in array]
    else: 
        str_list = [str(a) for a in array]
        
    return splitter.join(str_list)

def read_mni_obj(filename):
    '''
    Read MNI obj file
    output:
    header_info: list of surfprop, n_points, nitems, colour_flag, colour_table, end_indices
    point_array: np array of points
    normals: np array of normals
    faces: np array of faces
    '''
    with open(filename, 'r') as f:
        data = f.readlines()

        s = '\n'.join(data)
        data = s.split()

        if data[0] != 'P':
            raise ValueError('Only Polygons supported')

        surfprop = [float(data[1]), float(data[2]), float(data[3]), int(data[4]), int(data[5])]
        n_points = int(data[6])

        start = 7
        end = n_points * 3 + start
        point_array = [np.float32(x) for x in data[start:end]]
        point_array = np.reshape(point_array, (n_points, 3,))

        start = end
        end = n_points * 3 + start
        normals = [np.float32(x) for x in data[start:end]]
        normals = np.reshape(normals, (n_points, 3,))

        nitems = int(data[end])
        colour_flag = int(data[end + 1])

        start = end + 2
        end = start + 4
        colour_table = np.array(data[start:end]).astype('float')

        start = end + 4
        end = start + 2*nitems

        start = end
        end = start + nitems
        end_indices = [int(i) for i in data[start:end]]

        start = end
        end = start + end_indices[-1] + 1
        indices = [int(i) for i in data[start:end]]
        faces = np.reshape(indices, (-1, 3,))

        header_info = surfprop, n_points, nitems, colour_flag, colour_table, end_indices
        
    return header_info, point_array, normals, faces


def save_mni_obj(filename, header_info, point_array, normals, faces, overwrite=False):
    """
    Write this object to a file.
    """
    print(f'Writing to {filename}')
    if not overwrite:
        assert not os.path.exists(filename), f'Do not overwrite {filename}!'
    
    surfprop, n_points, nitems, colour_flag, colour_table, end_indices = header_info
    
    with open(filename, 'w') as file:
#         float(data[1]), float(data[2]), float(data[3]), int(data[4]), int(data[5])
        header = ['P'] + surfprop + [n_points]
        file.write(list2str(header, two_deci=False) + '\n')

        for point in point_array:
            file.write('  ' + list2str(point, splitter='  ') + '\n')

        for vector in normals:
            file.write(' ' + list2str(vector) + '\n')

        file.write(f'\n{nitems}\n')

        colour_str = ' '.join(list(colour_table.astype('str'))) + '\n'
        colour_str = colour_str*(n_points)
        file.write(f'{colour_flag} {colour_str}\n')
        
        
        for i in range(0, nitems, 8):
            file.write(' ' + list2str(end_indices[i:i + 8], two_deci=False) + '\n')

#         for i in range(0, len(faces), 8):
#             file.write(' ' + list2str(faces.flatten()) + '\n')
        file.write(f'\n')
        file.write(list2str(faces.flatten(), two_deci=False) + '\n')
