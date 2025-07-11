import trimesh
import time
import igl
import dipy.reconst.shm as shm
import dipy.core.sphere as dsp
import numpy as np

def interp_vertex_val_to_face(vertex, faces, data, method='mean'):
    """
    vertex: 3D numpy array, shape (N, 3)
    faces: 3D numpy array, shape (M, 3)
    data: 1D numpy array, shape (N,)
    """
    v_data1 = data[faces[:,0]]
    v_data2 = data[faces[:,1]]
    v_data3 = data[faces[:,2]]
    
    p1 = vertex[faces[:,0]]
    p2 = vertex[faces[:,1]]
    p3 = vertex[faces[:,2]]

    centers = (p1 + p2 + p3) / 3

    if method == 'mean':
        return (v_data1 + v_data2 + v_data3) / 3
    elif method == 'nn':
        dist = np.linalg.norm(np.array([p1-centers, p2-centers, p3-centers]),axis=2)
        nn = np.argmin(dist, axis=0)
        idx = faces[np.arange(len(faces)), nn]
        return data[idx]

def interp_face_val_to_vertex(vertex, faces, data, method='mean'):
    """
    vertex: 2D numpy array, shape (N, 3)
    faces: 2D numpy array, shape (M, 3)
    data: 1D numpy array, shape (M,)
    """
    if method == 'mean':
        sum_vertex = np.zeros((len(vertex),), dtype=data.dtype)
        count_vertex = np.zeros((len(vertex),), dtype=int)
        for i, face in enumerate(faces):
            for v in face:
                sum_vertex[v] += data[i]
                count_vertex[v] += 1
        vertex_data = sum_vertex / count_vertex
        return vertex_data
    elif method == 'nn':
        assert False, 'Not implemented yet'

def mean_over_mesh(vertex, faces, data):
    """
    vertex: 3D numpy array, shape (N, 3)
    faces: 3D numpy array, shape (M, 3)
    data: 1D numpy array, shape (M,)
    """
    vec1 = vertex[faces[:,1]] - vertex[faces[:,0]]
    vec2 = vertex[faces[:,2]] - vertex[faces[:,0]]
    cross = np.cross(vec1, vec2, axis=1)
    areas = np.linalg.norm(cross, axis=1) / 2
    return np.sum(areas * data) / np.sum(areas)

def is_in_mesh(p, V=None, F=None, mesh=None):
    '''
    inputs:
        p: point_number*3 
        V: vertex_number*3
        F: face_number*3
    return:
        point_number*1 bool
    '''
    if mesh is None:
        mesh = trimesh.Trimesh(vertices=V, faces=F)
    start = time.time()
    is_contained = mesh.contains(p)
    end = time.time()
    print('time:', end-start)
    return is_contained

def point_mesh_dist(P, V, F):
    '''
    inputs:
        P: point_number*3
        V: vertex_number*3
        F: face_number*3
    return:
        point_number*1 float
    '''
    start = time.time()
    D, I, C = igl.point_mesh_squared_distance(P, V, F)
    end = time.time()
    print('time:', end-start)
    return D, I, C

def Matrix_B(SV, max_order=8):
    sphere = dsp.Sphere(xyz=SV)
    B, B_inv = shm.sh_to_sf_matrix(sphere, sh_order=max_order, basis_type=None, full_basis=False, legacy=True, return_inv=True, smooth=0)
    cur_idx = 1
    new_B = np.zeros_like(B.T)
    new_B[:,0] = B.T[:,0]
    for n in [5, 9, 13, 17]:
        new_B[:,cur_idx:cur_idx+n] = B.T[:,cur_idx+n-1:cur_idx-1:-1]
        cur_idx += n
    return new_B

def get_cortex_mask(V_wm, N_wm, F_wm, V_pial, N_pial, F_pial, vol_shape, pial_offset, wm_offset, h):
    P = get_cortex_pts(V_wm, N_wm, F_wm, V_pial, N_pial, F_pial, vol_shape, pial_offset, wm_offset, h)
    idx = np.rint(P/h).astype(np.int)
    mask = np.zeros(vol_shape)
    for idx_p in idx:
        mask[tuple(idx_p)] = 1
    return mask

def get_cortex_pts(V_wm, N_wm, F_wm, V_pial, N_pial, F_pial, vol_shape, pial_offset, wm_offset, h):
    """
    V_wm: (n,3)
    F_wm: (m,3)
    V_pial: (n,3)
    F_pial: (m,3)
    vol_shape: (3,)
    h: float
    return: (n,3)
    """

    P = get_all_pts(vol_shape, h)
    P_in_pial = filter_pts_by_mesh(P, V_pial, N_pial, F_pial, dist_offset = pial_offset, inside=True)
    P_cortex = filter_pts_by_mesh(P_in_pial, V_wm, N_wm, F_wm, dist_offset = wm_offset, inside=False)
    return P_cortex

def get_all_pts(vol_shape, h):
    """
    vol_shape: (3,)
    h: float
    return: (n,3)
    """
    x = np.arange(0, vol_shape[0]*h, h)
    y = np.arange(0, vol_shape[1]*h, h)
    z = np.arange(0, vol_shape[2]*h, h)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    P = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)], axis=1)
    return P.astype(np.float32)

def filter_pts_by_mesh(P, V, N, F, dist_offset=0, inside=True):
    """
    P: (n,3)
    V: (m,3)
    N: (m,3)
    F: (l,3)
    dist_threshold: float
    inside: bool
    if inside return sdf < dist_offset, else return sdf > dist_offset
    return: (k,3)
    """
    SDF, _, _ = igl.signed_distance(P, V, F)
    if inside:
        mask = SDF < dist_offset
    else:
        mask = SDF > dist_offset

    P = P[mask]
    return P
    dist_mask = D < dist_threshold

    #threshold the distance
    D = D[dist_mask]
    I = I[dist_mask]
    C = C[dist_mask]
    P = P[dist_mask]

    Proj_F = F[I]

    V1 = V[Proj_F[:,0]]
    V2 = V[Proj_F[:,1]]
    V3 = V[Proj_F[:,2]]

    B = igl.barycentric_coordinates_tri(C, V1, V2, V3)
    N_P = N[Proj_F[:,0]]*B[:,0][:,None] + N[Proj_F[:,1]]*B[:,1][:,None] + N[Proj_F[:,2]]*B[:,2][:,None]
    #filter inside point
    inside_mask = np.sum(N_P*(C-P), axis=1) > 0
    if inside:
        P = P[inside_mask]
    else:
        P = P[~inside_mask]
    return P

# def gen_surf_mask(mask, data_folder, subj, hemi, h, shrink_dis = 0.5):
#     _, vertex, n, _ = load_wm_mesh(data_folder, subj, hemi)
#     shrinked_V = vertex - n*shrink_dis
#     for v in shrinked_V:
#         idx_x, idx_y, idx_z = floor(v[0]/h), floor(v[1]/h), floor(v[2]/h)
#         for x in [idx_x, idx_x+1]:
#             for y in [idx_y, idx_y+1]:
#                 for z in [idx_z, idx_z+1]:
#                     if x < mask.shape[0] and y < mask.shape[1] and z < mask.shape[2]:
#                         mask[x, y, z] = 1
#     return mask