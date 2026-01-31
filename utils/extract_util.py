from .mesh_util import Matrix_B
import numpy as np
from .data_io import read_mni_obj
import igl
import cvxpy as cp
from dipy.direction import peaks as Pk
from dipy.core.sphere import Sphere

def sorted_peaks(FOD_data, sphere_V, sphere_F, max_peaks=5, mask=None, relative_peak_threshold=0.1, min_separation_angle=10, peak_threshold=0.01):

    dipy_sph = Sphere(xyz=sphere_V, faces=sphere_F)
    dipy_sph.edges = dipy_sph.edges.astype(np.uint16)

    B = Matrix_B(sphere_V, max_order = 8)
    shape = FOD_data.shape
    peaks = np.zeros((shape[0], shape[1], shape[2], max_peaks, 3))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if mask is not None and mask[i,j,k] == 0:
                    continue
                v = FOD_data[i,j,k,0:45]
                y = B @ v
                peak_dir, peak_amp, peak_idx_ = Pk.peak_directions(y, dipy_sph, relative_peak_threshold=relative_peak_threshold, min_separation_angle=min_separation_angle, is_symmetric=True)
                
                amp_mask = peak_amp > peak_threshold
                peak_dir = peak_dir[amp_mask]
                peak_amp = peak_amp[amp_mask]
                peak_idx_ = peak_idx_[amp_mask]

                if len(peak_amp) == 0:
                    continue

                if len(peak_amp) > max_peaks:
                    peak_idx_ = peak_idx_[0:max_peaks]
                    peak_dir = peak_dir[0:max_peaks]
                    peak_amp = peak_amp[0:max_peaks]
                peaks[i,j,k,0:len(peak_idx_),:] = peak_dir*peak_amp[:,np.newaxis]
    return peaks

def generate_glyph_mesh(vertices, faces, fod_values):
    """Scale vertices based on FOD values and return new vertices and faces."""
    scaled_vertices = vertices * fod_values[:, np.newaxis]
    return scaled_vertices, faces

def adjacency_list(F):
    '''
    F: (n_F*3) face list
    '''
    n_V = F.max() + 1
    adj_list = [set() for _ in range(n_V)]
    for f in F:
        adj_list[f[0]].update({f[1], f[2]})
        adj_list[f[1]].update({f[0], f[2]})
        adj_list[f[2]].update({f[0], f[1]})
    return adj_list

def pair_peaks(peak_idx_sorted):
    peak_pairs = []
    if len(peak_idx_sorted)==0:
        return peak_pairs

    assert len(peak_idx_sorted)%2 == 0
    n_peaks = int(len(peak_idx_sorted)/2)
    for i in range(n_peaks):
        peak_pairs.append([peak_idx_sorted[2*i], peak_idx_sorted[2*i+1]])
    return peak_pairs

def find_peak_idx(y, sphere_adj_list, threshold=0.01, relative_threshold = None):
    '''
    y: (n_SV*1) y = BS*v 
    '''
    y[y<0] = 0
    peak_idx = []
    for i,y_ in enumerate(y):
        if relative_threshold is not None and y_ < relative_threshold*np.max(y):
            continue

        if y_ < threshold:
            continue
        neighbors = sphere_adj_list[i]

        if y_ > np.max([y[n] for n in neighbors]):
            peak_idx.append(i)

    return peak_idx

def peak_idx_sorted(y, sym_pairs, sphere_adj_list, threshold=0.01, relative_threshold = None):
    peak_idx = find_peak_idx(y, sphere_adj_list, threshold=threshold, relative_threshold=relative_threshold)

    peak_idx_op = [sym_pairs[i] for i in peak_idx]
    peak_idx = list(set(peak_idx + peak_idx_op))
    sorted_idx = np.argsort(-y[peak_idx]) # sort the peak_idx by y
    peak_idx = np.array(peak_idx)[sorted_idx]

    return peak_idx

def Matrix_A(B, idx1, idx2, adj_list, num_ring=2):
    '''
    get the matrix A for the idx vertice.
    matrix A is the num_ring neighbor of the idx vertice.
    '''
    ring_idx = n_ring_idx(idx1, adj_list, num_ring)
    ring_idx_op = n_ring_idx(idx2, adj_list, num_ring)
    mask = np.ones(B.shape[0], dtype=bool)
    mask[list(ring_idx)] = False
    mask[list(ring_idx_op)] = False
    A = B.copy()
    A[mask,:] = 0
    return A

def check_symmetry(V):
    sym_pairs = {}
    for i,v1 in enumerate(V):
        flag = False
        for j,v2 in enumerate(V):
            if np.linalg.norm(v1 + v2) < 1e-3:
                flag = True
                break
        if not flag:
            print('Not symmetric')
            return
        else:
            sym_pairs[i] = j
    return sym_pairs

def n_ring_idx(center_idx, adj_list, num_ring=2):
    '''
    get the n-ring index of the center_idx.
    '''
    ring_idx = set([center_idx])
    for _ in range(num_ring):
        new_ring_idx = set()
        for idx in ring_idx:
            new_ring_idx |= adj_list[idx]
        ring_idx |= new_ring_idx
    return ring_idx

def get_normal_vertices(V, N, degree=np.pi/3):
    """
    return the idx of normal vertices, degree must be in (0, pi/2)
    """
    assert degree < np.pi/2
    assert degree > 0
    return np.argwhere(V @ N > np.cos(degree)).flatten()

def get_tangent_vertices(V, N, degree=np.pi/3):
    """
    return the idx of tangent vertices, degree must be in (0, pi/2)
    """
    assert degree < np.pi/2
    assert degree > 0
    return np.argwhere(np.abs(V @ N) < np.cos(degree)).flatten()

def get_tan_plane_from_rad(rad_directions, peak_directions, peak_amplitude, tan_angle=np.pi/6, mask=None):
    '''
    find the tangential plane. If 1 tan peak is find inside the tan_angle threshold, define the tangent plane normal direction by rad_directions - rad_directions_projected_on_tan_peak_direction

    rad_direction: (x,y,z,3) radial peak directions
    peak_directions: (x,y,z,n,3) peak directions
    peak_amplitude: (x,y,z,n) peak amplitude
    tan_angle: tangent angle threshold
    return: (x,y,z,3) tangent plane normal directions
    '''
    shape = rad_directions.shape[0:3]

    assert peak_directions.shape[0:3] == shape
    assert peak_amplitude.shape[0:3] == shape

    compute_mask = np.linalg.norm(rad_directions, axis=-1) > 0
    if mask is not None:
        compute_mask = compute_mask * mask
    
    #threshold the peak directions by tan_angle
    dot_prod = (rad_directions[...,None,:]*peak_directions).sum(-1)

    tan_mask = np.logical_and(np.abs(dot_prod) < np.sin(tan_angle), np.abs(dot_prod) > 0)
    tan_mask = tan_mask * compute_mask[...,None]

    #voxels with no tan peak
    null_mask = (peak_amplitude * tan_mask).sum(-1) < 1e-5 #(x,y,z)
    tan_peak_idx = np.argmax(peak_amplitude * tan_mask, axis=-1) #(x,y,z)
    tan_peak_dir = peak_directions[np.arange(shape[0])[:,None,None], np.arange(shape[1])[None,:,None], np.arange(shape[2])[None,None,:], tan_peak_idx] #(x,y,z,3)
    tan_peak_dir = tan_peak_dir/(np.linalg.norm(tan_peak_dir, axis=-1)[...,None]+1e-7)
    tan_peak_dir[null_mask] = 0

    #find the tangent plane normal
    tan_normal = np.zeros((shape[0], shape[1], shape[2], 3))
    for i,j,k in np.ndindex(shape):
        if compute_mask[i,j,k] and not null_mask[i,j,k]:
            rad_dir = rad_directions[i,j,k]
            tan_dir = tan_peak_dir[i,j,k]
            tan_normal[i,j,k] = rad_dir - (rad_dir @ tan_dir) * tan_dir
            tan_normal[i,j,k] /= np.linalg.norm(tan_normal[i,j,k])

    return tan_normal

def get_rad_tan_voxel(voxel_spharms, rad_direction, tan_normal, B, SV, SF, mass_matrix, threshold=0.01, rad_angle=np.pi/6, tan_angle=np.pi/6):
    '''

    '''
    if np.linalg.norm(rad_direction) == 0 and np.linalg.norm(tan_normal) == 0:
        return 0, 0
    
    amplitude = B @ voxel_spharms
    amplitude[amplitude < threshold] = 0
    if np.linalg.norm(rad_direction) == 0:
        rad_comp = 0
    else:
        rad_mask = (SV @ rad_direction) > np.cos(rad_angle)
        rad_comp = mass_matrix.dot(amplitude * rad_mask).sum()

    if np.linalg.norm(tan_normal) == 0:
        tan_comp = 0
    else:
        tan_mask = np.abs(SV @ tan_normal) < np.sin(tan_angle)
        tan_comp = mass_matrix.dot(amplitude * tan_mask).sum()

    return rad_comp, tan_comp

def get_rad_tan(FOD, rad_dir_vol, tan_normal_vol, compute_mask, threshold, rad_angle=np.pi/6, tan_angle=np.pi/6):
    _, SV, _, SF = read_mni_obj('/ifs/loni/faculty/shi/spectrum/Student_2020/hzhang/data/ADNI/scripts/sphere5120.obj')
    B = Matrix_B(SV, max_order=8)
    mass_matrix = igl.massmatrix(SV, SF, igl.MASSMATRIX_TYPE_VORONOI)
    
    vol_shape = rad_dir_vol.shape[0:3]
    assert vol_shape == FOD.shape[0:3]
    tan_comp = np.zeros(vol_shape)
    rad_comp = np.zeros(vol_shape)
    for i,j,k in np.ndindex(vol_shape):
        rad_dir = rad_dir_vol[i,j,k,:]
        tan_normal = tan_normal_vol[i,j,k,:]
        if compute_mask[i,j,k] == 0:
            continue
        voxel_spharms = FOD[i,j,k,:]
        ver, tan = get_rad_tan_voxel(voxel_spharms, rad_dir, tan_normal, B, SV, SF, mass_matrix, threshold=threshold, rad_angle=rad_angle, tan_angle=tan_angle)
        rad_comp[i,j,k] = ver
        tan_comp[i,j,k] = tan
    return rad_comp, tan_comp 

def decompose_peak_voxel(B, v, sym_pairs, adj_list, threshold=0.01, relative_threshold = None, num_ring=3, lambda_1=1, lambda_2=0.01, lambda_3=1):
    amplitude = B @ v
    amplitude[amplitude < 0] = 0
    peak_idx = peak_idx_sorted(amplitude, sym_pairs, adj_list, threshold=threshold, relative_threshold=relative_threshold)
    if len(peak_idx)==0:
        return None

    assert len(peak_idx)%2 == 0
    n_peaks = int(len(peak_idx)/2)
    A = np.zeros((n_peaks, B.shape[0], B.shape[1]))
    for k in range(n_peaks):
        idx1 = peak_idx[2*k]
        idx2 = peak_idx[2*k+1]
        A[k] = Matrix_A(B, idx1, idx2, adj_list, num_ring=num_ring)

    #cvxpy optimization
    U = cp.Variable((n_peaks, B.shape[1]))
    recon_cost = cp.sum_squares(cp.sum(U, axis=0) - v)

    local_cost = 0
    for k in range(n_peaks):
        local_cost += lambda_1*cp.sum_squares(A[k] @ (U[k] - v))

    single_peak_cost = 0
    for k in range(n_peaks):
        single_peak_cost += lambda_2*cp.sum_squares((B - A[k])@ U[k])

    supre_cost = 0
    for k in range(n_peaks):
        in_norm_term = 0
        for j in range(n_peaks):
            if j != k:
                in_norm_term += A[j] @ U[k]
        supre_cost += lambda_3*cp.sum_squares(in_norm_term)


    objective = cp.Minimize(recon_cost + local_cost + single_peak_cost + supre_cost)
    cp.Problem(objective).solve(solver=cp.SCS, verbose=False)
    return U.value

def decompose_radial_peak(B, v, normal_direction, SV, sym_pairs, adj_list, threshold=0.01, num_ring=3, lambda_1=1, lambda_2=0.01, lambda_3=1):
    amplitude = B @ v
    amplitude[amplitude < 0] = 0
    peak_idx = peak_idx_sorted(amplitude, sym_pairs, adj_list, threshold=threshold)

    assert len(peak_idx)%2 == 0

    if len(peak_idx)==0:
        return None
    
    #find radial index
    peak_idx_ = peak_idx[0:len(peak_idx):2]#combine opposite direction to 1
    assert len(peak_idx_) > 0
    peak_dir = SV[peak_idx_]
    peak_frac = amplitude[peak_idx_]
    cos_d_angle = np.abs(peak_dir @ normal_direction)

    rad_idx = np.argmax(cos_d_angle)
    if cos_d_angle[rad_idx] < np.cos(np.pi/6):
        return None
    elif len(peak_idx) == 2: #only one peak and the peak is in normal direction
        return v

    #cvxpy optimization
    n_peaks = int(len(peak_idx)/2)
    assert n_peaks > 1, 'number of peaks insufficient for peak decomposition'


    Ak = np.zeros((B.shape[0], B.shape[1]))
    
    idx1 = peak_idx[2*rad_idx]
    idx2 = peak_idx[2*rad_idx+1]
    Ak = Matrix_A(B, idx1, idx2, adj_list, num_ring=num_ring)


    U = cp.Variable((2, B.shape[1]))
    recon_cost = cp.sum_squares(cp.sum(U, axis=0) - v)

    local_cost = 0
    local_cost += lambda_1*cp.sum_squares(Ak @ (U[0] - v))

    single_peak_cost = 0
    single_peak_cost += lambda_2*cp.sum_squares((B - Ak)@ U[0])

    objective = cp.Minimize(recon_cost + local_cost + single_peak_cost)
    cp.Problem(objective).solve(solver = cp.SCS, verbose=False)

    return U.value[0]

def get_tan_rad_dir_from_peak(rad_peak, tan_peak, normal):
    '''
    rad_peak: (x,y,z,3) radial peak
    tan_peak: (x,y,z,3) tangential peak
    normal: (x,y,z,3) normal direction

    return:
    rad_dir_vol (x,y,z,3) radial peak direction
    tan_plane_normal_vol (x,y,z,3) tangent plane normal direction
    '''
    rad_mask = np.linalg.norm(rad_peak, axis=-1) > 0
    tan_mask = np.linalg.norm(tan_peak, axis=-1) > 0
    tan_peak_normed = tan_peak/(np.linalg.norm(tan_peak, axis=-1)[...,None] + 1e-8)

    # tan_and_rad_mask = np.logical_and(tan_mask, rad_mask)
    only_tan_mask = np.logical_and(tan_mask, np.logical_not(rad_mask))
    only_rad_mask = np.logical_and(rad_mask, np.logical_not(tan_mask))

    rad_dir_vol = rad_peak/(np.linalg.norm(rad_peak, axis=-1)[...,None] + 1e-8)
    rad_dir_vol[only_tan_mask] = normal[only_tan_mask]

    print(rad_dir_vol.shape, tan_peak_normed.shape)

    rad_proj_tan = (rad_dir_vol*tan_peak_normed).sum(-1)[...,None] * tan_peak_normed
    tan_plane_normal_vol = rad_dir_vol - rad_proj_tan
    tan_plane_normal_vol = tan_plane_normal_vol/(np.linalg.norm(tan_plane_normal_vol, axis=-1)[...,None] + 1e-8)

    tan_plane_normal_vol[only_rad_mask] = 0
    rad_dir_vol[only_tan_mask] = 0

    return rad_dir_vol, tan_plane_normal_vol

        



