import numpy as np
import random
from scipy import optimize
import math
import pickle
import os



def estimate(xyzs):
    u, s, v = np.linalg.svd(xyzs)
    normal = v[3:].T
    error = s[3]
    return normal, error

def fit_plane_iterative(xyzs):
    # this function takes 3D points
    # in the shape of 4xn
    # u should be 4x1
    # scipy's least_square function utilized for plane fitting
    print(xyzs.shape)
    def plane_cost(u):
        nonlocal xyzs
        u = np.reshape(u, (-1, 1))
        # print("xyzs shape is ", xyzs.shape)
        # print("u shape is ", u.shape)
        return np.sum(np.dot(xyzs.T, u)**2) + 5*(np.linalg.norm(u) - 1)**2

    u0 = np.random.rand(4)
    u_sol = optimize.least_squares(plane_cost, u0)
    return u_sol.x, u_sol.cost


def fit_plane_mean(xyzs, normal_pre):
    xyz = [data[:3] for data in xyzs]
    # xyz = np.concatenate((xyz, normal_pre), axis=0)
    xyz.append(normal_pre)
    mean_pt = np.mean(xyz, axis=0)
    xyz_tilde = xyz - mean_pt
    u, s, v = np.linalg.svd(xyz_tilde)
    normal = np.reshape(v[-1], (-1,1))
    error = s[2]
    mean = -np.mean(mean_pt)
    return normal, error, mean

def ad_hoc_normals(xyzs, it):
    normals = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])]
    center_pt = np.mean(xyzs[:3, :], axis=1)
    mean = -1.0*np.mean(center_pt)
    m = np.zeros((4,1))
    # m[3,:] = mean
    if it == 0 or it == 2:
        m[:3,:] = normals[0]
        m[3,:] = -np.dot(normals[0].T, center_pt)
    elif it == 1 or it == 3:
        m[:3,:] = normals[1]
        m[3, :] = -np.dot(normals[1].T, center_pt)
    else:
        m[:3,:] = normals[2]
        m[3, :] = -np.dot(normals[2].T, center_pt)
    return m, 0



def is_inlier(coeffs, xyz, threshold):
    return np.abs(np.dot(coeffs.T, xyz)) < threshold

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_planes(v1, v2):
    """

    :param v1: 4x1 vector of plane 1
    :param v2: 4x1 vector of plane 2
    :return: the angle between two planes
    """
    epsilon = 1-6
    v1 = v1[:3]
    v2 = v2[:3]
    ratio = np.dot(v1.T, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    ratio = np.clip(ratio, -1 + epsilon, 1 - epsilon)
    ratio = np.abs(ratio)
    return np.arccos(ratio)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations,
               threshold=0.4, stop_at_goal=True, random_seed=None, mode="svd"):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data.T)
    normal_pre = np.zeros((3,))
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        if mode == "mean":
            n_, err, mean = estimate(s, normal_pre)
            m = np.zeros((4, 1))
            m[:3, :] = n_
            m[3, :] = mean
        else:
            m, err = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j], threshold):
                ic += 1

        # print(s)
        # print('estimate:', m,)
        # print('# inliers:', ic)
        if(i%100 is 0):
            print("runsac step: ", i)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    print('Total number of points: ', len(data))
    return best_model, best_ic

def run_ransac_rect(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, rotation_matrices,
               threshold=0.4, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    best_error = 1500
    best_orto_error = 1500
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    total_data = sum([plane_coord_hom.shape[1] for plane_coord_hom in data])
    goal_inliers_no = total_data*goal_inliers
    print("total data no is ", total_data)

    for i in range(max_iterations):
        selected_coord_hom = []
        for plane_coord_hom in data:
            plane_coord_hom_list = list(plane_coord_hom.T)
            s = random.sample(plane_coord_hom_list, int(sample_size))
            selected_coord_hom.append(np.array(s).T)

        m, err = estimate(selected_coord_hom, rotation_matrices)
        n1 = m[0][:3, :] / np.linalg.norm(m[0][:3, :])
        n2 = m[1][:3, :] / np.linalg.norm(m[1][:3, :])
        n3 = m[4][:3, :] / np.linalg.norm(m[4][:3, :])
        # N_mat should be 3x3
        N_mat = np.concatenate((n1, n2, n3), axis=1)
        error_mat = np.eye(3) - np.dot(N_mat.T, N_mat)
        error = np.trace(np.dot(error_mat.T, error_mat))

        ic = 0
        for data_, normal in zip(data, m):
            data_ = list(data_.T)
            for point in data_:
                if is_inlier(normal, point, threshold):
                    ic += 1

        # print(s)
        # print('estimate:', m,)
        # print('# inliers:', ic)
        if(i%100 is 0):
            print("runsac step: ", i)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            best_error = err
            best_orto_error = error
            if ic > goal_inliers_no and stop_at_goal:
                break
    print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    print('Total number of points with rect: ', len(data))
    print('Orthogonality error is: ', best_orto_error)
    return best_model, best_ic, best_error

def run_ransac_rect_ver_2(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, rotation_matrices,
               threshold=0.4, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    best_error = 1500
    best_ortho_error = 1500
    random.seed(random_seed)
    eps = 1e-5
    orthonogonal_found = False
    # random.sample cannot deal with "data" being a numpy array
    total_data = sum([plane_coord_hom.shape[1] for plane_coord_hom in data])
    goal_inliers_no = total_data*goal_inliers
    print("total data no is ", total_data)

    for i in range(max_iterations):
        selected_coord_hom = []
        for plane_coord_hom in data:
            plane_coord_hom_list = list(plane_coord_hom.T)
            s = random.sample(plane_coord_hom_list, int(sample_size))
            selected_coord_hom.append(np.array(s).T)

        m, err = estimate(selected_coord_hom, rotation_matrices)
        n1 = m[0][:3,:] / np.linalg.norm(m[0][:3,:])
        n2 = m[1][:3,:] / np.linalg.norm(m[1][:3,:])
        n3 = m[4][:3, :] / np.linalg.norm(m[4][:3, :])
        # N_mat should be 3x3
        N_mat = np.concatenate((n1, n2, n3), axis=1)
        error_mat = np.eye(3) - np.dot(N_mat.T, N_mat)
        error = np.trace(np.dot(error_mat.T, error_mat))

        ic = 0
        for data_, normal in zip(data, m):
            data_ = list(data_.T)
            for point in data_:
                if is_inlier(normal, point, threshold):
                    ic += 1

        # print(s)
        # print('estimate:', m,)
        # print('# inliers:', ic)
        if(i%100 is 0):
            print("runsac step: ", i)

        if(ic > best_ic) and (error < eps):
            best_ic = ic
            best_model = m
            best_error = err
            best_ortho_error = error
            orthonogonal_found = True
            if ic > goal_inliers_no and stop_at_goal:
                break

        if (ic > best_ic) and not orthonogonal_found:
            best_ic = ic
            best_model = m
            best_error = err
            best_ortho_error = error
            if ic > goal_inliers_no and stop_at_goal:
                break
    print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    print('Total number of points with rect: ', len(data))
    print('Orthogonality error is: ', best_ortho_error)
    return best_model, best_ic, best_error, best_ortho_error

def point_dist_to_plane(point, normal):
    """
    This function calculates the shortest distance from a point to
    a plane
    :param point: 4x1 np array, x,y,z and 1
    :param normal: 4x1 np array, plane vector
    :return: shortest distance from point to normal
    """

    return np.abs(np.dot(point.T, normal))/np.linalg.norm(normal[:3,0])

def save_geometry_off(corners, window_pts, door_pts, name_to_save):
    faces = ['4 0 6 4 2', '4 0 6 7 1', '4 6 4 5 7', '4 4 2 3 5', '4 2 0 1 3',
             '4 3 1 7 5', '4 8 9 10 11', '4 12 13 14 15', '4 16 17 18 19']
    with open(name_to_save, 'w') as f:
        f.write('OFF \n 20 9 0 \n')
        np.savetxt(f, corners, delimiter=' ', fmt='%6f %6f %6f')
        np.savetxt(f, window_pts[0][:3,:].T, delimiter=' ', fmt='%6f %6f %6f')
        np.savetxt(f, door_pts[0][:3, :].T, delimiter=' ', fmt='%6f %6f %6f')
        np.savetxt(f, window_pts[1][:3, :].T, delimiter=' ', fmt='%6f %6f %6f')
        for face in faces:
            f.write(face + '\n')

def save_geometry_off_2(corners, window_pts, door_pts, name_to_save):
    faces = ['4 0 6 4 2', '4 0 6 7 1', '4 6 4 5 7', '4 4 2 3 5', '4 2 0 1 3',
             '4 3 1 7 5', '4 8 9 10 11', '4 12 13 14 15']
    with open(name_to_save, 'w') as f:
        f.write('OFF \n 16 8 0 \n')
        np.savetxt(f, corners, delimiter=' ', fmt='%6f %6f %6f')
        np.savetxt(f, window_pts[0][:3,:].T, delimiter=' ', fmt='%6f %6f %6f')
        np.savetxt(f, door_pts[0][:3, :].T, delimiter=' ', fmt='%6f %6f %6f')
        # np.savetxt(f, window_pts[1][:3, :].T, delimiter=' ', fmt='%6f %6f %6f')
        for face in faces:
            f.write(face + '\n')

def take_proj_onto_plane(pts, normal):
    n = normal[:3,0]/np.linalg.norm(normal[:3,0])
    proj = np.eye(3) - np.dot(n, n.T)
    pts_3 = pts[:3,:]
    pts_3_proj = np.dot(proj, pts_3)
    pts[:3,:] = pts_3_proj

    return pts

# def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
#     '''
#     The Iterative Closest Point estimator.
#     Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
#     their relative pose and the number of iterations
#     Returns the affine transform that transforms
#     the cloudpoint a to the cloudpoint b.
#     Note:
#         (1) This method works for cloudpoints with minor
#         transformations. Thus, the result depents greatly on
#         the initial pose estimation.
#         (2) A large number of iterations does not necessarily
#         ensure convergence. Contrarily, most of the time it
#         produces worse results.
#     '''
#
#     src = np.array([a.T], copy=True).astype(np.float32)
#     dst = np.array([b.T], copy=True).astype(np.float32)
#
#     #Initialise with the initial pose estimation
#     Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
#                    [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
#                    [0,                    0,                   1          ]])
#
#     src = cv2.transform(src, Tr[0:2])
#
#     for i in range(no_iterations):
#         #Find the nearest neighbours between the current source and the
#         #destination cloudpoint
#         nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto',
#                                 warn_on_equidistant=False).fit(dst[0])
#         distances, indices = nbrs.kneighbors(src[0])
#
#         #Compute the transformation between the current source
#         #and destination cloudpoint
#         T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
#         #Transform the previous source and update the
#         #current source cloudpoint
#         src = cv2.transform(src, T)
#         #Save the transformation from the actual source cloudpoint
#         #to the destination
#         Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
#     return Tr[0:2]

def find_rotation_from_two_vectors(a, b):
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    v = np.cross(a, b)
    v = np.reshape(v, (-1,))
    s = np.linalg.norm(v) + 1e-4
    c = np.dot(a.T, b)
    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + v_x + np.dot(v_x, v_x)*(1 - c)/(s**2)

    return R

def find_all_homografies(normals1, normals2):
    homografies = []
    for normal1, normal2 in zip(normals1, normals2):
        homografies.append(find_rotation_from_two_vectors(normal1[:3,0], normal2[:3,0]))
    return homografies

def calculate_angle_error(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)

    b_norm = [b_[:3, 0] / np.linalg.norm(b_[:3, 0]) for b_ in b]

    n_gt = [np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
            np.array([0, 0, -1])]

    angle_errors = [180*np.arccos(abs(np.dot(b_, n_)))/np.pi for b_, n_ in zip(b_norm, n_gt)]

    return angle_errors

def InterpolationSearch(fun):
    thetas = [-5+i*0.01 for i in range(int(10/0.01))]
    lys = [fun(theta) for theta in thetas]
    min_index = lys.index(min(lys))
    return thetas[min_index]

def get_tetrahedron_vol(v1, v2, v3):
    # vi's should be (3,1)
    return abs(np.dot(np.cross(v2.T,v3.T), v1) / 6.0)

def vol_calculator_box(corners):
    center_corn = np.mean(np.array(corners), axis=0)
    corners = [corner - center_corn for corner in corners]
    vol = 0
    for i in range(8):
        vol += get_tetrahedron_vol(corners[i % 8], corners[(i + 1) % 8], corners[(i + 2) % 8])

    vol += get_tetrahedron_vol(corners[0], corners[2], corners[4]) + \
           get_tetrahedron_vol(corners[2], corners[4], corners[6])
    vol += get_tetrahedron_vol(corners[1], corners[3], corners[5]) + \
           get_tetrahedron_vol(corners[3], corners[5], corners[7])

    return vol


def compute_rect_w_method_4_5(plane_coord_list_hom):
    normals = []
    errors = []
    # plane_coord_list_hom = [plane_coord.T for plane_coord in plane_coord_list_hom]

    plane_coord_list = []
    plane_center_list = []
    for plane_coord in plane_coord_list_hom:
        plane_center_list.append(np.mean(plane_coord[:3, :], axis=1))
        plane_coord_list.append(plane_coord[:3, :] - np.reshape(np.mean(plane_coord[:3, :], axis=1), (-1, 1)))
        # has shape of 3xN

    # augmented_correlation = np.zeros((3, 3))
    # mean_point_number = 0
    # for P_mat, R_mat in zip(plane_coord_list, rotation_matrices):
    #     # P_mat is 3xN
    #     # R_mat is 3x3
    #     augmented_correlation += self.compute_correlation(P_mat, R_mat)
    #     mean_point_number += P_mat.shape[1] / len(plane_coord_list)

    A_mat = np.dot(plane_coord_list[0], plane_coord_list[0].T) + np.dot(plane_coord_list[2], plane_coord_list[2].T)
    B_mat = np.dot(plane_coord_list[1], plane_coord_list[1].T) + np.dot(plane_coord_list[3], plane_coord_list[3].T)
    C_mat = np.dot(plane_coord_list[4], plane_coord_list[4].T) + np.dot(plane_coord_list[5], plane_coord_list[5].T)

    # normals_list = [n2_opt, n1, n2_opt, n1, n3_opt, n3_opt]
    N0 = rotation_matrix([0, 0, 1], 0.2 * np.pi)
    # N0 = np.reshape(N0, (-1,))

    def fun(n):
        # n = np.reshape(n, (3,3))
        nonlocal A_mat, B_mat, C_mat
        penalty_param = 1.0
        i1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        i2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        i3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        mat_1 = np.dot(n.T, np.dot(A_mat, np.dot(n, i1.T)))
        mat_2 = np.dot(n.T, np.dot(B_mat, np.dot(n, i2.T)))
        mat_3 = np.dot(n.T, np.dot(C_mat, np.dot(n, i3.T)))
        return np.trace(mat_1) + np.trace(mat_2) + np.trace(mat_3) + \
               penalty_param * np.linalg.norm(np.dot(n, n.T) - np.eye(3), ord='fro')

    def jac(n):
        nonlocal A_mat, B_mat, C_mat
        i1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        i2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        i3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

        mat_1 = np.dot(A_mat, np.dot(n, i1))
        mat_2 = np.dot(B_mat, np.dot(n, i2))
        mat_3 = np.dot(C_mat, np.dot(n, i3))

        return 2 * (mat_1 + mat_2 + mat_3)

    def my_minimize(fun, jac, n0, max_iter=100, epsilon=1e-6):
        n = n0
        for i in range(max_iter):
            cost = fun(n)
            if cost < epsilon:
                break
            cost_der = jac(n)
            W = np.dot(cost_der, n.T) - np.dot(n, cost_der.T)

            def y_theta(x):
                nonlocal W, n, fun
                mat_1 = np.linalg.inv(np.eye(3) + (x/2)*W)
                mat_2 = np.eye(3) - (x/2)*W
                return np.dot(mat_1, np.dot(mat_2, n))

            def my_fun(x):
                nonlocal y_theta
                return fun(y_theta(x))

            theta_opt = InterpolationSearch(my_fun)
            n = y_theta(theta_opt)

        return n

    normal_mat = my_minimize(fun, jac, N0, max_iter=100, epsilon=1e-8)

    # res = minimize(fun, N0, method='Nelder-Mead', tol=1e-6)
    # normal_mat = np.reshape(res.x, (3,3))

    n1 = np.reshape(normal_mat[:, 0], (-1, 1))
    n2 = np.reshape(normal_mat[:, 1], (-1, 1))
    n3 = np.reshape(normal_mat[:, 2], (-1, 1))

    normals_list = [n1, n2, n1, n2, n3, n3]

    for normal, center in zip(normals_list, plane_center_list):
        curr_normal = np.zeros((4, 1))
        curr_normal[:3, :] = np.reshape(np.array(normal), (-1, 1))
        curr_normal[3, :] = -np.dot(curr_normal[:3, :].T, center)
        normals.append(curr_normal)

    for normal_, plane_coord in zip(normals, plane_coord_list_hom):
        # erros_mat is Nx1
        errors_mat = np.dot(plane_coord.T, normal_) / np.reshape(np.linalg.norm(plane_coord.T[:, :3], axis=1),
                                                                 (-1, 1))
        errors.append(np.dot(errors_mat.T, errors_mat))

    return normals, errors

def calculate_gt_vertex_error(corners, gt_corners):
    """
    This functions calculates the minimum square error between two
    boxes with 8 corners with a rigid body transformation between them
    :param corners: Corner points of the model to asses -- np array in shape (8,3)
    :param gt_corners: Corner points of the ground truth model -- np array in shape (8,3)
    :return: the least square error distance between corners of the model and the ground truth
    """

    corner_mean = np.mean(corners, axis=0, keepdims=True) # [1,3]
    gt_mean = np.mean(gt_corners, axis=0, keepdims=True) # [1,3]

    normalized_x = (corners - corner_mean).T # [3,8]
    normalized_y = (gt_corners - gt_mean).T # [3,8]

    scatter_matrix = np.dot(normalized_x, normalized_y.T) # [3,3]
    u,s,vh = np.linalg.svd(scatter_matrix)
    error_term_1 = np.trace(np.dot(normalized_x.T,normalized_x))
    error_term_2 = np.trace(np.dot(normalized_y.T,normalized_y))
    error_term_3 = np.sum(s)

    return  error_term_1 + error_term_2 - 2*error_term_3

def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - np.mean(model, axis=1, keepdims=True)
    data_zerocentered = data - np.mean(data, axis=1, keepdims=True)

    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh
    # print(model)
    # print(model.mean(1))
    # scale = np.trace(np.dot(np.dot(model_zerocentered.T, rot), data_zerocentered)) / np.trace(np.dot(model_zerocentered.T, model_zerocentered))
    scale = np.sqrt(np.sum(data_zerocentered**2) / np.sum(model_zerocentered**2))
    trans = np.mean(data, axis=1, keepdims=True) - scale * rot * np.mean(model, axis=1, keepdims=True)

    model_aligned = scale * np.dot(rot, model) + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]

    return rot,trans,scale,trans_error

def save_obj(obj, name, folder):
    with open(os.path.join(folder, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, folder ):
    with open(os.path.join(folder, name + '.pkl'), 'rb') as f:
        return pickle.load(f)



