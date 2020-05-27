import numpy as np
import os, glob
import cv2
from PIL import Image
import random
from scipy.optimize import minimize
from plyProcess.plyMain import plyProcess
from plyProcess.ransac import estimate, point_dist_to_plane, run_ransac, is_inlier, \
    fit_plane_mean, ad_hoc_normals, rotation_matrix, find_all_homografies, run_ransac_rect, run_ransac_rect_ver_2, \
    InterpolationSearch, align
from collections import defaultdict



class thermal:
    def __init__(self, param):
        self.main_root = param['root']
        self.root = param['root'] + "/outside_out_pmvs/reconstruction_sequential/PMVS/"
        self.thermal_root = param["thermal_root"]
        self.image_root = param["image_root"]
        self.ply_pro = plyProcess(self.root + "models/pmvs_options.txt.ply")
        self.box_method = param['box_method']
        # self.ply_pro = plyProcess(self.root + "models/sahin.ply")

        # let's list the thermal images and sort the list
        # self.thermal_images = glob.glob(self.thermal_root + "/*.jpg")
        # self.thermal_images.sort(reverse=False)
        self.thermal_images, self.electroptic_images = self.find_thermal_list()
        self.window_3d_coordinates = []
        self.door_3d_coordinates = []

        # let's list the electroptic images and sort the list
        # self.electroptic_images = glob.glob(self.root + "visualize/*.jpg")
        # self.electroptic_images.sort(reverse=False)

        # let's add the intrinsic and extrinsic parameters for FLIR E60
        self.rgb_intrinsic = np.loadtxt(param['rgb_intrinsic'], delimiter=',')
        self.thermal_intrinsic = np.loadtxt(param['thermal_intrinsic'], delimiter=',')
        self.rot_diff_mat = np.loadtxt(param['rot_diff_mat'], delimiter=',')
        self.trans_diff_ = np.loadtxt(param['trans_diff_'], delimiter=',')

        # find the projection matrices
        self.projectionMatrices = glob.glob(self.root + 'txt/*')
        self.projectionMatrices.sort()

        self.iterationNo = param['iterationNo']
        self.begin_it = param['begin_it']
        self.max_iterations = 1000
        self.ransac_threshold = param['ransac_threshold']
        self.mean_temps_pixel = []
        self.penalty_param = 1.0

    def only_read_ply(self, method_selection):
        # read the point cloud file
        self.ply_ = self.ply_pro.read_ply_file(method_selection)
        return self.ply_

    def change_penalty_param(self, penalty_param):
        self.penalty_param = penalty_param
        print("Penalty parameter change to: ", self.penalty_param)

    def change_electrooptic_images(self, electroptic_images):
        self.electroptic_images = electroptic_images

    def add_thermal_points(self, method_selection, thermal_image_no=0):
        """
        This function takes the point cloud and add it the thermal points
        from the selected thermal image using the rotation and translation
        :param thermal_image_no:  this should be an int to
        specifiy the image number
        :return: a numpy array corresponding to a thermal point cloud
        """
        # read the selected thermal image and convert it into RGB
        self.thermal_image_to_add = cv2.imread(self.thermal_images[thermal_image_no])
        # read the point cloud file

        self.ply = self.ply_pro.read_ply_file(method_selection)

        # read projection matrice
        def read_projection_matrix(projection_matrix_no):
            with open(self.projectionMatrices[projection_matrix_no], 'r') as f:
                projection = f.readlines()

            projection.remove(projection[0])
            projection = list(map(lambda a: a.split(), projection))
            projection = np.float64(projection)

            return projection

        self.projection = read_projection_matrix(thermal_image_no)

        # to get rotation and translation
        self.rot_trans = np.dot(np.linalg.pinv(self.rgb_intrinsic), self.projection)
        # split the matrix into rotation and translation
        self.rot_, self.trans_ = self.rot_trans[:, 0:3], self.rot_trans[:, 3]
        self.rot_the = np.dot(self.rot_diff_mat, self.rot_)
        self.trans_the = np.dot(self.rot_diff_mat, self.trans_) + self.trans_diff_
        # put together new rotation and translation
        self.rot_trans_the = np.concatenate((self.rot_the, np.reshape(self.trans_the, (-1, 1))), axis=1)

        # projection matrix of thermal
        self.project_the = np.dot(self.thermal_intrinsic, self.rot_trans)
        # add thermal points to the point cloud
        self.ply_ = self.ply_pro.add_thermal_points(self.project_the, self.thermal_image_to_add)

        return self.ply_

    def add_thermal_points_all(self, method_selection):
        """

        This function takes the point cloud and add it the thermal points
        from all the thermal images using the rotation and translation
        :param thermal_image_no:  this should be an int to
        specifiy the image number
        :return: a numpy array corresponding to a thermal point cloud
        """
        # read the selected thermal image and convert it into RGB

        # read the point cloud file
        self.ply = self.ply_pro.read_ply_file(method_selection)

        for i, thermal_img in enumerate(self.thermal_images):
            self.thermal_image_to_add = cv2.imread(thermal_img)
            thermal_img_to_add_gray = cv2.cvtColor(self.thermal_image_to_add, cv2.COLOR_BGR2GRAY)
            mean_thermal = np.mean(np.mean(self.thermal_image_to_add))
            self.mean_temps_pixel.append(mean_thermal)

            # read projection matrice
            self.projection = self.read_projection_matrix(i)

            # to get rotation and translation
            self.rot_trans = np.dot(np.linalg.pinv(self.rgb_intrinsic), self.projection)
            # split the matrix into rotation and translation
            self.rot_, self.trans_ = self.rot_trans[:, :3], self.rot_trans[:, 3]
            self.rot_the = np.dot(self.rot_diff_mat, self.rot_)
            self.trans_the = np.dot(self.rot_diff_mat, self.trans_) + self.trans_diff_
            # put together new rotation and translation
            self.rot_trans_the = np.concatenate((self.rot_the, np.reshape(self.trans_the, (-1, 1))), axis=1)

            # projection matrix of thermal
            self.project_the = np.dot(self.thermal_intrinsic, self.rot_trans)
            # add thermal points to the point cloud
            self.ply_ = self.ply_pro.add_thermal_points(self.project_the, self.thermal_image_to_add)

        return self.ply_

    def save_processed_point_cloud(self, name_to_save):
        # self.ply_pro.save_point_cloud(self.root + "models/" + name_to_save + ".ply")
        self.ply_pro.save_point_cloud_lib(self.root + "models/" + name_to_save + ".ply")

    def find_thermal_list(self):

        try:
            with open(self.root + '/list.txt', 'r') as f:
                list_file = f.readlines()
        except:
            list_file = os.listdir(os.path.join(self.root, 'visualize'))

        list_file = [a.split() for a in list_file]
        file_names = [x[0] for x in list_file]
        # print(file_names)

        electro_file = os.listdir(self.image_root)
        electro_file.sort(reverse=False)

        # arr_mask = [bool(x in file_names) for x in electro_file]
        arr_mask = [True for x in electro_file]

        thermal_file = os.listdir(self.thermal_root)
        thermal_file.sort(reverse=False)

        selected_thermal = []
        for i, x in enumerate(thermal_file):
            if (arr_mask[i]):
                selected_thermal.append(os.path.join(self.thermal_root, x))

        selected_electroptic = []
        for i, x in enumerate(electro_file):
            if (arr_mask[i]):
                selected_electroptic.append(os.path.join(self.image_root, x))

        print("Thermal images")
        print(selected_thermal)
        print("Electroptic Images")
        print(selected_electroptic)

        return selected_thermal, selected_electroptic

    def window_selection_from_points(self, img_window_list=[], mode="window"):

        if mode == "door":
            self.selected_pt_door = []
        elif mode == "point":
            self.select_pt_point = defaultdict()
        else:
            self.selected_pt_win = []

        def click_and_crop(event, x, y, flags, param):
            nonlocal ref_pt, cur_images
            if event == cv2.EVENT_LBUTTONDOWN:
                ref_pt.append((x, y))
                # self.image_to_select = cv2.circle(self.image_to_select, (x, y), 10, (0, 0, 255), thickness=2)
                cur_images.append(cv2.circle(cur_images[-1], (x, y), 10, (0, 0, 255), thickness=2))
                cv2.imshow(self.windowName, cur_images[-1])
                val_ = int(len(ref_pt) / 4)
                if (len(ref_pt) >= 4) and (len(ref_pt) % 4 is 0):
                    ctr = np.array(ref_pt[-4:]).reshape((-1, 1, 2)).astype(np.int32)
                    self.image_to_select = cv2.drawContours(self.image_to_select, [ctr], -1,
                                                            (0, 255, 0), thickness=2)
                    cv2.imshow(self.windowName, cur_images[-1])
                print(x, y)
                # print("Number of images: {}".format(len(cur_images)))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(cur_images) > 1:
                    # del ref_pt[-1], cur_images[-1]
                    cur_images.append(cv2.circle(cur_images[-1], (x, y), 10, (0, 0, 255), thickness=2))
                    cv2.imshow(self.windowName, cur_images[-1])
                    print("Number of images: {}".format(len(cur_images)))
                else:
                    print("One than one points should be selected for undo")

        count_click = 0
        img_shapes = []
        for i, no in enumerate(img_window_list):
            # check if the window image changes
            if (i is 0):
                self.image_to_select = cv2.imread(self.electroptic_images[no])
            elif (img_window_list[i] != img_window_list[i - 1]):
                self.image_to_select = cv2.imread(self.electroptic_images[no])
            self.windowName = 'img number ' + str(no) + '/' + str(len(self.electroptic_images))
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image_to_select, 'Please click the four corners of the window',
                        (20, 80), self.font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
            if mode == "point":
                cv2.resizeWindow(self.windowName, int(1024*1.5), int(768*1.5))
            else:
                cv2.resizeWindow(self.windowName, 1024, 768)
            # u_ratio, v_ratio = self.image_to_select.shape[0] / 1024, self.image_to_select.shape[1] / 768
            cv2.imshow(self.windowName, self.image_to_select)
            ref_pt = []
            cur_images = [self.image_to_select]
            cv2.setMouseCallback(self.windowName, click_and_crop)

            k = cv2.waitKey(0)
            if k % 256 == 27:  # wait for ESC key to exit
                cv2.destroyAllWindows()
                # break
            elif k % 256 == ord('a'):  # wait for 's' key to save and exit
                # print("ref_pt is ", ref_pt)
                ref_pt_ord = self.order_the_selected_points(ref_pt)
                if mode == "door":
                    self.selected_pt_door.append(ref_pt_ord)
                elif mode == "point":
                    self.select_pt_point[no] = ref_pt
                else:
                    self.selected_pt_win.append(ref_pt_ord)
                img_window_list.insert(i + 1, no)
                # print("Before the destroy command")
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
            # elif k % 256 == ord('r'):  # wait for 's' key to save and exit
            #     if len(cur_images) > 1:
            #         del ref_pt[-1], cur_images[-1]
            #         cv2.imshow(self.windowName, cur_images[-1])
            #     else:
            #         print("One than one points should be selected for undo")
            elif k % 256 == ord('q'):  # wait for 's' key to save and exit
                ref_pt_ord = self.order_the_selected_points(ref_pt)
                print("ref_pt is ", ref_pt_ord)
                if mode == "door":
                    self.selected_pt_door.append(ref_pt_ord)
                elif mode == "point":
                    self.select_pt_point[no] = ref_pt
                else:
                    self.selected_pt_win.append(ref_pt_ord)
                # self.ply_pro.exclude_window_points(self.read_projection_matrix(no), ref_pt_ord,
                #                                    self.image_to_select.shape)
                # print("Before the destroy command")
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
            else:
                raise Exception("Invalid Selection!, k value is {}".format(k))

        # print("points are ", self.selected_pt_win)

    def door_selection_from_points(self, img_door_list=[]):

        self.selected_pt_door = []

        def click_and_crop(event, x, y, flags, param):
            nonlocal ref_pt
            nonlocal u_ratio, v_ratio
            if event == cv2.EVENT_LBUTTONDOWN:
                ref_pt.append((x * u_ratio, y * v_ratio))
                if len(ref_pt) >= 4:
                    ctr = np.array(ref_pt).reshape((-1, 1, 2)).astype(np.int32)
                    self.image_to_select = cv2.drawContours(self.image_to_select, [ctr], -1,
                                                            (0, 255, 0), thickness=2)
                    cv2.imshow(self.windowName, self.image_to_select)
                print(x, y)

        count_click = 0
        for i, no in enumerate(img_door_list):
            # self.image_to_select = cv2.imread(self.electroptic_images[no])
            if (i is 0):
                self.image_to_select = cv2.imread(self.electroptic_images[no])
            elif (img_door_list[i] != img_door_list[i - 1]):
                self.image_to_select = cv2.imread(self.electroptic_images[no])
            self.windowName = 'img number ' + str(no) + '/' + str(len(self.electroptic_images))
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image_to_select, 'Please click the four corners of the window',
                        (20, 80), self.font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.windowName, 1024, 768)
            u_ratio, v_ratio = self.image_to_select.shape[0] / 1024, self.image_to_select.shape[1] / 768
            cv2.imshow(self.windowName, self.image_to_select)
            ref_pt = []
            cv2.setMouseCallback(self.windowName, click_and_crop)

            # ref_pt_ord = self.order_the_selected_points(ref_pt)

            k = cv2.waitKey(0)
            if k % 256 == 27:  # wait for ESC key to exit
                cv2.destroyAllWindows()
                break
            elif k % 256 == ord('a'):  # wait for 's' key to save and exit
                # print("ref_pt is ", ref_pt)
                ref_pt_ord = self.order_the_selected_points(ref_pt)
                self.selected_pt_door.append(ref_pt_ord)
                img_door_list.insert(i + 1, no)
                # print("Before the destroy command")
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
            elif k % 256 == ord('q'):  # wait for 's' key to save and exit
                # print("ref_pt is ", ref_pt)
                ref_pt_ord = self.order_the_selected_points(ref_pt)
                self.selected_pt_door.append(ref_pt_ord)
                # self.ply_pro.exclude_window_points(self.read_projection_matrix(no), ref_pt_ord,
                #                                    self.image_to_select.shape)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
            else:
                print("Invalid Selection!")

        print("points are ", self.selected_pt_door)

    def find_3_pt_from_pixel(self, img_window_list=[]):

        self.window_3d_coordinates = []
        for i, no in enumerate(img_window_list):
            # read the corresponding projection matrix
            self.projection = self.read_projection_matrix(no)
            # print("Projection is ", self.projection)
            # current_window = self.selected_pt_win[i]
            U = np.array(self.selected_pt_win[i])
            self.window_3d_coordinates.append(
                self.ply_pro.get_selected_window_pt(self.projection, self.selected_pt_win[i]))
            # print("3D windows are \n", self.window_3d_coordinates[i])

        return self.window_3d_coordinates

    def find_3_pt_from_pixel_and_projection(self, normals, img_window_list=[], normal_no=[],
                                            mode="window"):

        self.window_3d_coordinates = []
        for i, no in enumerate(img_window_list):
            # read the corresponding projection matrix
            self.projection = self.read_projection_matrix(no)
            # print("Projection is ", self.projection)
            # current_window = self.selected_pt_win[i]
            # self.projection = np.dot(homografies[normal_no], self.projection)
            # U has a shape of (4,2)
            if (mode == "door"):
                U = np.array(self.selected_pt_door[i])
            else:
                U = np.array(self.selected_pt_win[i])
            # U1 has shape (4,4)
            U1 = np.concatenate((U.T, np.ones((1, 4)), np.zeros((1, 4))), axis=0)
            n1 = np.reshape(normals[normal_no[i]], (1, -1))
            A = np.concatenate((self.projection, n1), axis=0)
            Xs = np.dot(np.linalg.pinv(A), U1)
            # mean = np.mean(Xs[:3,:], axis=0)
            # Xs[:3,:] = np.dot(homografies[normal_no], (Xs[:3,:] - mean)) + mean
            Xs = Xs / Xs[3, :]
            self.window_3d_coordinates.append(Xs)
            # print("3D windows are \n", self.window_3d_coordinates[i])

        return self.window_3d_coordinates

    def find_3_pt_from_pixel_door(self, img_door_list=[]):

        self.door_3d_coordinates = []
        for i, no in enumerate(img_door_list):
            # read the corresponding projection matrix
            self.projection = self.read_projection_matrix(no)
            # print("Projection is ", self.projection)

            self.door_3d_coordinates = self.ply_pro.get_selected_window_pt(self.projection, self.selected_pt_door[i])
            print("3D windows are \n", self.door_3d_coordinates)
        return self.door_3d_coordinates

    def assign_wall_to_window(self, window_3d_coord, no_walls):
        """
        this assigns object to nearest wall
        :param window_3d_coord: list of np arrays
        :return: numberOfWindows, list
        """
        number_of_windows = [0 for i in range(no_walls)]
        for point_3d in window_3d_coord:
            max_dists = np.zeros((no_walls, 1))
            for j in range(no_walls):
                for i in range(4):
                    point = point_3d[:, i]
                    normal = self.normals[j]
                    dist = point_dist_to_plane(point, normal)
                    if (dist > max_dists[j, 0]):
                        max_dists[j, 0] = dist

            win_place = np.argmin(max_dists)
            number_of_windows[(win_place + 1) % len(no_walls)] += 1
        return number_of_windows

    def thermal_image_point_select(self, box_method=1, ransac_param=0.1, plane_list=[], method=2, mode=1):
        # self.ply = self.ply_pro.read_ply_file()

        self.selected_pt = []

        def click_and_crop(event, x, y, flags, param):
            nonlocal ref_pt
            if event == cv2.EVENT_LBUTTONDOWN:
                ref_pt.append((x, y))
                if len(ref_pt) >= 2:
                    self.image_to_select = cv2.rectangle(self.image_to_select, ref_pt[0], ref_pt[1],
                                                         (0, 255, 0), thickness=2)
                    cv2.imshow(self.windowName, self.image_to_select)
                print(x, y)

        count_click = 0
        for i, no in enumerate(plane_list):
            self.image_to_select = cv2.imread(self.electroptic_images[no])
            self.windowName = 'img number ' + str(no) + '/' + str(len(self.electroptic_images))
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image_to_select, 'Please click the two corners of the rectangle',
                        (20, 80), self.font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.windowName, 1024, 768)
            cv2.imshow(self.windowName, self.image_to_select)
            ref_pt = []
            cv2.setMouseCallback(self.windowName, click_and_crop)
            self.selected_pt.append(ref_pt)
            k = cv2.waitKey(0)
            if k % 256 == 27:  # wait for ESC key to exit
                cv2.destroyAllWindows()
                # break
            elif k % 256 == ord('q'):  # wait for 's' key to save and exit
                print("q is pressed")
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
            else:
                print("Invalid Selection!, k is {}".format(k))
        print("points are ", self.selected_pt)

        # normals1, errors1, mean_temps1 = self.thermal_plane_fitting(plane_list=plane_list, method=method, mode=mode)
        # normals2, errors2, mean_temps2 = self.thermal_rectangle_prism_fitting(box_method, ransac_param, plane_list=plane_list,
        #                                                                       method=method)
        # homografies = find_all_homografies(normals1, normals2)
        #
        # return normals2, errors2, mean_temps2, homografies
        # return normals1, errors1, mean_temps1, homografies

    def thermal_plane_fitting(self, tol=0.5, plane_list=[], method=2, mode=1):

        """

        :param tol: tolerance value for ransac, scalar
        :param plane_list: list of image numbers for plane fitting
        :param method: selection for only 3D or 3D with thermal
        :param mode: if set to 2 a rectangle is fit
        :return: list of normals as np arrays with shape (4,1),
            list of errors, list of mean temperature values
        """

        self.normals = []
        self.errors = []
        self.mean_temps = []
        self.mean_temps_2 = []

        for i, no in enumerate(plane_list):
            # read the corresponding projection matrix
            self.projection = self.read_projection_matrix(no)
            # print("Projection is ", self.projection)

            self.image = cv2.imread(self.electroptic_images[no])
            # self.plane_coordinates, self.mean_temp = self.ply_pro.get_selected_coord_for_plane(
            #     self.image, self.projection, self.selected_pt[no]
            # )
            self.plane_coordinates, self.mean_temp = self.ply_pro.get_selected_coord_for_plane(
                self.image, self.projection, self.selected_pt[i]
            )

            # normal_pre = np.zeros((4, 1))
            # if i != 0 and i != 5:
            #     normal_pre[:3,:] = self.normals[i-1][:3,:]

            # self.plane_coordinates = np.concatenate((self.plane_coordinates, normal_pre), axis=1)

            # print("projection ", no, "is ", self.plane_coordinates[:,:10])
            # c = np.mean(self.plane_coordinates, axis=1, keepdims=True)
            # m, b = estimate(self.plane_coordinates[:, :2000])
            # m, b = fit_plane_iterative(self.plane_coordinates)
            # print(self.plane_coordinates[:, 152])
            # print("The mean point is ", c)
            goal_inliers = int(self.plane_coordinates.shape[1] * 0.7)
            # print("mean temp is ", self.mean_temp)
            if mode is 2:
                # m, b = run_ransac(self.plane_coordinates,
                #                   fit_plane_mean, lambda x, y, tol: is_inlier(x, y, tol),
                #                   20, goal_inliers, self.max_iterations, threshold=self.ransac_threshold, mode="mean")
                m, b = ad_hoc_normals(self.plane_coordinates, i)
            else:
                m, b = run_ransac(self.plane_coordinates,
                                  estimate, lambda x, y, tol: is_inlier(x, y, tol),
                                  20, goal_inliers, self.max_iterations, threshold=self.ransac_threshold)

            self.normals.append(m)
            self.errors.append(b)
            if (method is 2):
                self.mean_temps.append(self.mean_temp)
                self.mean_temps_2.append(self.mean_temps_pixel[no])

        # print("Mean pixel temps are\n", self.mean_temps_pixel)
        return self.normals, self.errors, self.mean_temps

    def thermal_rectangle_prism_fitting(self, box_method, ransac_param, tol=0.5, plane_list=[], method=2):

        """

        :param tol: tolerance value for ransac, scalar
        :param plane_list: list of image numbers for plane fitting
        :param method: selection for only 3D or 3D with thermal
        :param mode: if set to 2 a rectangle is fit
        :return: list of normals as np arrays with shape (4,1),
            list of errors, list of mean temperature values
        self.plane_coordinates is a np array with shape (4xN) augmented with 1's
        """

        self.normals = []
        self.errors = []
        self.mean_temps = []
        self.mean_temps_2 = []
        self.plane_coord_list = []
        self.plane_center_list = []
        self.plane_coord_hom_list = []
        # self.rotation_matrices = [np.eye(3), rotation_matrix([0,0,1], -0.5*np.pi), rotation_matrix([0,0,1], np.pi),
        #                           rotation_matrix([0,0,1], 0.5*np.pi), rotation_matrix([1,0,0], 0.5*np.pi),
        #                           rotation_matrix([1,0,0], -0.5*np.pi)]

        self.rotation_matrices = [np.eye(3), rotation_matrix([0, 0, 1], -0.5 * np.pi),
                                  rotation_matrix([0, 0, 1], np.pi),
                                  rotation_matrix([0, 0, 1], 0.5 * np.pi), rotation_matrix([0, 1, 0], 0.5 * np.pi),
                                  rotation_matrix([0, 1, 0], -0.5 * np.pi)]

        for i, no in enumerate(plane_list):
            # read the corresponding projection matrix
            self.projection = self.read_projection_matrix(no)
            # print("Projection is ", self.projection)

            self.image = cv2.imread(self.electroptic_images[no])
            # self.plane_coordinates, self.mean_temp = self.ply_pro.get_selected_coord_for_plane(
            #     self.image, self.projection, self.selected_pt[no]
            # )
            self.plane_coordinates, self.mean_temp = self.ply_pro.get_selected_coord_for_plane(
                self.image, self.projection, self.selected_pt[i]
            )

            self.plane_center_list.append(np.mean(self.plane_coordinates[:3, :], axis=1))
            self.plane_coord_list.append(self.plane_coordinates[:3, :] - np.reshape(self.plane_center_list[i], (-1, 1)))
            self.plane_coord_hom_list.append(self.plane_coordinates)

            if (method is 2):
                self.mean_temps.append(self.mean_temp)
                self.mean_temps_2.append(self.mean_temps_pixel[no])

        # augmented_correlation = np.zeros((3,3))
        # mean_point_number = 0
        # for P_mat, R_mat in zip(self.plane_coord_list, self.rotation_matrices):
        #     # P_mat is 3xN
        #     # R_mat is 3x3
        #     augmented_correlation += self.compute_correlation(P_mat, R_mat)
        #     mean_point_number += P_mat.shape[1] / len(self.plane_coord_list)
        #
        #
        # w, v = np.linalg.eigh(augmented_correlation)
        #
        # for R_mat, center in zip(self.rotation_matrices, self.plane_center_list):
        #     curr_normal = np.zeros((4,1))
        #     curr_normal[:3,:] = np.reshape(np.dot(R_mat, v[:,0]), (-1,1))
        #     curr_normal[3,:] = -np.dot(curr_normal[:3,:].T, center)
        #     self.normals.append(curr_normal)
        #
        # self.errors = np.sqrt(w[0]) / mean_point_number
        selected_coord_hom = []
        for plane_coord_hom in self.plane_coord_hom_list:
            plane_coord_hom_list = list(plane_coord_hom.T)
            s = random.sample(plane_coord_hom_list, 5)
            selected_coord_hom.append(np.array(s).T)

        self.normals, self.errors, self.ortho_errors = self.compute_rect_w_ransac(box_method, ransac_param, self.plane_coord_hom_list,
                                                               self.rotation_matrices)

        # print("Mean pixel temps are\n", self.mean_temps_pixel)
        return self.normals, self.errors, self.mean_temps, self.ortho_errors

    def compute_correlation(self, P_mat, R_mat):
        # P_corr = np.dot(P_mat, P_mat.T)
        return np.dot(np.dot(R_mat.T, np.dot(P_mat, P_mat.T)), R_mat)

    def compute_rect_w(self, plane_coord_list_hom, rotation_matrices):
        normals = []

        plane_coord_list = []
        plane_center_list = []
        for plane_coord in plane_coord_list_hom:
            plane_center_list.append(np.mean(plane_coord[:3, :], axis=1))
            plane_coord_list.append(plane_coord[:3, :] - np.reshape(np.mean(plane_coord[:3, :], axis=1), (-1, 1)))

        augmented_correlation = np.zeros((3, 3))
        mean_point_number = 0
        for P_mat, R_mat in zip(plane_coord_list, rotation_matrices):
            # P_mat is 3xN
            # R_mat is 3x3
            augmented_correlation += self.compute_correlation(P_mat, R_mat)
            mean_point_number += P_mat.shape[1] / len(plane_coord_list)

        w, v = np.linalg.eigh(augmented_correlation)

        for R_mat, center in zip(rotation_matrices, plane_center_list):
            curr_normal = np.zeros((4, 1))
            curr_normal[:3, :] = np.reshape(np.dot(R_mat, v[:, 0]), (-1, 1))
            curr_normal[3, :] = -np.dot(curr_normal[:3, :].T, center)
            normals.append(curr_normal)

        error = w[0]
        return normals, error

    def compute_normals_6(self, plane_coord_list_hom, rotation_matrices):
        normals = []
        errors = []

        plane_coord_list = []
        plane_center_list = []
        for plane_coord in plane_coord_list_hom:
            plane_center_list.append(np.mean(plane_coord[:3, :], axis=1))
            plane_coord_list.append(plane_coord[:3, :] - np.reshape(np.mean(plane_coord[:3, :], axis=1), (-1, 1)))

        # augmented_correlation = np.zeros((3, 3))
        # mean_point_number = 0
        # for P_mat, R_mat in zip(plane_coord_list, rotation_matrices):
        #     # P_mat is 3xN
        #     # R_mat is 3x3
        #     augmented_correlation += self.compute_correlation(P_mat, R_mat)
        #     mean_point_number += P_mat.shape[1] / len(plane_coord_list)

        new_coordinates = [np.concatenate((plane_coord_list[0], plane_coord_list[2]), axis=1),
                           np.concatenate((plane_coord_list[1], plane_coord_list[3]), axis=1),
                           np.concatenate((plane_coord_list[4], plane_coord_list[5]), axis=1)]

        u1, s1, vh1 = np.linalg.svd(new_coordinates[0])
        n1 = np.reshape(u1[:, 2], (-1, 1))
        u2, s2, vh2 = np.linalg.svd(new_coordinates[1])
        n2 = np.reshape(u2[:, 2], (-1, 1))
        u3, s3, vh3 = np.linalg.svd(new_coordinates[2])
        n3 = np.reshape(u3[:, 2], (-1, 1))

        normals_list = [n1, n2, n1, n2, n3, n3]

        for normal_, center in zip(normals_list, plane_center_list):
            curr_normal = np.zeros((4, 1))
            curr_normal[:3, :] = np.reshape(normal_, (-1, 1))
            curr_normal[3, :] = -np.dot(curr_normal[:3, :].T, center)
            normals.append(curr_normal)

        for normal_, plane_coord in zip(normals, plane_coord_list_hom):
            # erros_mat is Nx1
            errors_mat = np.dot(plane_coord.T, normal_) / np.reshape(np.linalg.norm(plane_coord.T[:, :3], axis=1),
                                                                     (-1, 1))
            errors.append(np.dot(errors_mat.T, errors_mat))

        return normals, errors

    def compute_normals_6_n(self, plane_coord_list_hom, rotation_matrices):
        normals = []
        errors = []

        plane_coord_list = []
        plane_center_list = []
        for plane_coord in plane_coord_list_hom:
            plane_center_list.append(np.mean(plane_coord[:3, :], axis=1))
            plane_coord_list.append(plane_coord[:3, :] - np.reshape(np.mean(plane_coord[:3, :], axis=1), (-1, 1)))

        # augmented_correlation = np.zeros((3, 3))
        # mean_point_number = 0
        # for P_mat, R_mat in zip(plane_coord_list, rotation_matrices):
        #     # P_mat is 3xN
        #     # R_mat is 3x3
        #     augmented_correlation += self.compute_correlation(P_mat, R_mat)
        #     mean_point_number += P_mat.shape[1] / len(plane_coord_list)

        # new_coordinates = [np.concatenate((plane_coord_list[0], plane_coord_list[2]), axis=1),
        #                    np.concatenate((plane_coord_list[1], plane_coord_list[3]), axis=1),
        #                    np.concatenate((plane_coord_list[4], plane_coord_list[5]), axis=1)]
        #
        # u1, s1, vh1 = np.linalg.svd(new_coordinates[0])
        # n1 = np.reshape(u1[:, 2], (-1, 1))
        # u2, s2, vh2 = np.linalg.svd(new_coordinates[1])
        # n2 = np.reshape(u2[:, 2], (-1, 1))
        # u3, s3, vh3 = np.linalg.svd(new_coordinates[2])
        # n3 = np.reshape(u3[:, 2], (-1, 1))

        normals_list = []
        for plane_coord_ in plane_coord_list:
            u, s, vh = np.linalg.svd(plane_coord_)
            normals_list.append(np.reshape(u[:, 2], (-1, 1)))

        for normal_, center in zip(normals_list, plane_center_list):
            curr_normal = np.zeros((4, 1))
            curr_normal[:3, :] = np.reshape(normal_, (-1, 1))
            curr_normal[3, :] = -np.dot(curr_normal[:3, :].T, center)
            normals.append(curr_normal)

        for normal_, plane_coord in zip(normals, plane_coord_list_hom):
            # erros_mat is Nx1
            errors_mat = np.dot(plane_coord.T, normal_) / np.reshape(np.linalg.norm(plane_coord.T[:, :3], axis=1),
                                                                     (-1, 1))
            errors.append(np.dot(errors_mat.T, errors_mat))

        return normals, errors

    def compute_rect_w_other_method(self, plane_coord_list_hom, rotation_matrices):
        normals = []
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

        u, s, vh = np.linalg.svd(plane_coord_list[0])
        # a, b, c = u[:, 2]
        # normals_list = [[a, b, c], [-b, a, 0], [a, b, c], [-b, a, 0], [0, -c, b], [0, -c, b]]
        # print("a,b,c is ", u[:, 2])
        n1 = np.reshape(u[:, 2], (-1, 1))
        u2, s2, vh2 = np.linalg.svd(plane_coord_list[1])
        n2 = u2[:, 2]
        u3, s3, vh3 = np.linalg.svd(plane_coord_list[4])
        n3 = u3[:, 2]
        proj_a = np.dot(n1, np.dot(np.linalg.inv(np.dot(n1.T, n1)), n1.T))
        proj_a_ort = np.eye(proj_a.shape[0]) - proj_a
        n2_opt = np.dot(proj_a_ort, n2)
        n2_opt = n2_opt / np.linalg.norm(n2_opt)
        n3_opt = np.cross(n1, n2_opt, axis=0)
        n3_opt = n3_opt / np.linalg.norm(n3_opt)
        normals_list = [n1, n2_opt, n1, n2_opt, n3_opt, n3_opt]
        # normals_list = [n2_opt, n1, n2_opt, n1, n3_opt, n3_opt]

        for normal, center in zip(normals_list, plane_center_list):
            curr_normal = np.zeros((4, 1))
            curr_normal[:3, :] = np.reshape(np.array(normal), (-1, 1))
            curr_normal[3, :] = -np.dot(curr_normal[:3, :].T, center)
            normals.append(curr_normal)

        error = s[2]
        return normals, error

    def compute_rect_w_method_2(self, plane_coord_list_hom, rotation_matrices):
        normals = []
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

        normals_list = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[1], [0], [0]]),
                        np.array([[0], [1], [0]]), np.array([[0], [0], [1]]), np.array([[0], [0], [1]])]
        # normals_list = [n2_opt, n1, n2_opt, n1, n3_opt, n3_opt]

        for normal, center in zip(normals_list, plane_center_list):
            curr_normal = np.zeros((4, 1))
            curr_normal[:3, :] = np.reshape(np.array(normal), (-1, 1))
            curr_normal[3, :] = -np.dot(curr_normal[:3, :].T, center)
            normals.append(curr_normal)

        error = 0.1
        return normals, error

    def compute_rect_w_method_3(self, plane_coord_list_hom, rotation_matrices):
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

        eye_1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        eye_2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        eye_3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        # normals_list = [n2_opt, n1, n2_opt, n1, n3_opt, n3_opt]
        N0 = rotation_matrix([0, 0, 1], 0.2 * np.pi)

        def fun(n, a, b, c, i1, i2, i3):
            mat_1 = np.dot(n.T, np.dot(a, np.dot(n, i1.T)))
            mat_2 = np.dot(n.T, np.dot(b, np.dot(n, i2.T)))
            mat_3 = np.dot(n.T, np.dot(c, np.dot(n, i3.T)))
            return np.trace(mat_1) + np.trace(mat_2) + np.trace(mat_3)

        def jac(n, a, b, c, i1, i2, i3):
            mat_1 = np.dot(a, np.dot(n, i1))
            mat_2 = np.dot(b, np.dot(n, i2))
            mat_3 = np.dot(c, np.dot(n, i3))

            return 2 * (mat_1 + mat_2 + mat_3)

        def my_minimize(fun, jac, n0, a, b, c, i1, i2, i3, alpha=0.1, max_iter=100, epsilon=1e-4):
            n = n0
            for i in range(max_iter):
                cost = fun(n, a, b, c, i1, i2, i3)
                alpha = alpha / (i // 10 + 1)
                if cost < epsilon:
                    break
                cost_der = jac(n, a, b, c, i1, i2, i3)
                cost_der_proj = cost_der - np.dot(n, np.dot(cost_der.T, n))
                n = n - alpha * cost_der_proj
            return n

        normal_mat = my_minimize(fun, jac, N0, A_mat, B_mat, C_mat,
                                 eye_1, eye_2, eye_3, alpha=1e-2, max_iter=100, epsilon=1e-8)

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

    def compute_rect_w_method_4(self, plane_coord_list_hom, rotation_matrices):
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
        N0 = np.reshape(N0, (-1,))
        penalty_param = self.penalty_param

        def fun(n):
            n = np.reshape(n, (3,3))
            nonlocal A_mat, B_mat, C_mat
            nonlocal penalty_param
            i1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            i2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            i3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
            mat_1 = np.dot(n.T, np.dot(A_mat, np.dot(n, i1.T)))
            mat_2 = np.dot(n.T, np.dot(B_mat, np.dot(n, i2.T)))
            mat_3 = np.dot(n.T, np.dot(C_mat, np.dot(n, i3.T)))
            return np.trace(mat_1) + np.trace(mat_2) + np.trace(mat_3) + \
                   penalty_param * np.linalg.norm(np.dot(n, n.T) - np.eye(3), ord='fro')

        def jac(n, a, b, c, i1, i2, i3):
            mat_1 = np.dot(a, np.dot(n, i1))
            mat_2 = np.dot(b, np.dot(n, i2))
            mat_3 = np.dot(c, np.dot(n, i3))

            return 2 * (mat_1 + mat_2 + mat_3)

        def my_minimize(fun, jac, n0, a, b, c, i1, i2, i3, alpha=0.1, max_iter=100, epsilon=1e-4):
            n = n0
            for i in range(max_iter):
                cost = fun(n, a, b, c, i1, i2, i3)
                alpha = alpha / (i // 10 + 1)
                if cost < epsilon:
                    break
                cost_der = jac(n, a, b, c, i1, i2, i3)
                cost_der_proj = cost_der - np.dot(n, np.dot(cost_der.T, n))
                n = n - alpha * cost_der_proj
            return n

        # normal_mat = my_minimize(fun, jac, N0, A_mat, B_mat, C_mat,
        #                          eye_1, eye_2, eye_3, alpha=1e-2, max_iter=100, epsilon=1e-8)

        res = minimize(fun, N0, method='Nelder-Mead', tol=1e-6)
        normal_mat = np.reshape(res.x, (3,3))

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

    def compute_rect_w_method_5(self, plane_coord_list_hom, rotation_matrices):
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
        penalty_param = self.penalty_param

        def fun(n):
            # n = np.reshape(n, (3,3))
            nonlocal A_mat, B_mat, C_mat
            nonlocal penalty_param
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
                    mat_1 = np.linalg.inv(np.eye(3) + (x / 2) * W)
                    mat_2 = np.eye(3) - (x / 2) * W
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

    def compute_rect_w_method_baseline_and_4(self, plane_coord_list_hom, rotation_matrices):
        normals1, erros1 = self.compute_normals_6(plane_coord_list_hom, rotation_matrices)

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
        N0 = np.concatenate((normals1[0][:3, :], normals1[1][:3, :], normals1[4][:3, :]), axis=1)
        penalty_param = self.penalty_param
        # N0 = np.reshape(N0, (-1,))

        def fun(n):
            n = np.reshape(n, (3, 3))
            nonlocal A_mat, B_mat, C_mat
            nonlocal penalty_param
            i1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            i2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            i3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
            mat_1 = np.dot(n.T, np.dot(A_mat, np.dot(n, i1.T)))
            mat_2 = np.dot(n.T, np.dot(B_mat, np.dot(n, i2.T)))
            mat_3 = np.dot(n.T, np.dot(C_mat, np.dot(n, i3.T)))
            return np.trace(mat_1) + np.trace(mat_2) + np.trace(mat_3) + \
                   penalty_param * np.linalg.norm(np.dot(n, n.T) - np.eye(3), ord='fro')

        def jac(n, a, b, c, i1, i2, i3):
            mat_1 = np.dot(a, np.dot(n, i1))
            mat_2 = np.dot(b, np.dot(n, i2))
            mat_3 = np.dot(c, np.dot(n, i3))

            return 2 * (mat_1 + mat_2 + mat_3)

        # def my_minimize(fun, jac, n0, a, b, c, i1, i2, i3, alpha=0.1, max_iter=100, epsilon=1e-4):
        #     n = n0
        #     for i in range(max_iter):
        #         cost = fun(n, a, b, c, i1, i2, i3)
        #         alpha = alpha / (i // 10 + 1)
        #         if cost < epsilon:
        #             break
        #         cost_der = jac(n, a, b, c, i1, i2, i3)
        #         cost_der_proj = cost_der - np.dot(n, np.dot(cost_der.T, n))
        #         n = n - alpha * cost_der_proj
        #     return n

        # normal_mat = my_minimize(fun, jac, N0, A_mat, B_mat, C_mat,
        #                          eye_1, eye_2, eye_3, alpha=1e-2, max_iter=100, epsilon=1e-8)

        res = minimize(fun, N0, method='Nelder-Mead', tol=1e-6)
        normal_mat = np.reshape(res.x, (3, 3))

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


    def compute_rect_w_ransac(self, box_method, ransac_param, plane_coord_list_hom, rotation_matrices):

        def is_inlier(m, point, threshold):
            if abs(np.dot(m.T, point)) / np.linalg.norm(m[:3, 0]) < threshold:
                return True
            else:
                return False

        # compute_rect_w_other_method
        # compute_rect_w_method_2
        # compute_rect_w_method_3
        # compute_normals_6
        # compute_normals_6_n
        # compute_rect_w_method_baseline_and_4
        my_method = self.compute_normals_6_n
        if(box_method is 2):
            my_method = self.compute_rect_w_method_baseline_and_4

        best_m, best_ic, best_error, best_ortho_error = run_ransac_rect_ver_2(plane_coord_list_hom, my_method,
                                                            is_inlier, 4, 0.80, 100,
                                                            rotation_matrices, threshold=ransac_param,
                                                            stop_at_goal=True, random_seed=None)

        return best_m, best_error, best_ortho_error

    def compute_volume_corners(self, normals):
        corners = []
        for i in range(4):
            tmp_normal_matrix = np.concatenate((normals[i % 4], normals[(i + 1) % 4], normals[4]), axis=1)
            tmp_normal_matrix = tmp_normal_matrix.T
            corner = np.dot(np.linalg.pinv(tmp_normal_matrix[:, :3]), -1 * tmp_normal_matrix[:, 3])
            corners.append(corner)

            tmp_normal_matrix = np.concatenate((normals[i % 4], normals[(i + 1) % 4], normals[5]),
                                               axis=1)
            tmp_normal_matrix = tmp_normal_matrix.T
            corner = np.dot(np.linalg.pinv(tmp_normal_matrix[:, :3]), -1 * tmp_normal_matrix[:, 3])
            corners.append(corner)
        return corners

    def get_projection_matices(self):
        return self.projectionMatrices

    def order_the_selected_points(self, win_pt_2_order):
        if (win_pt_2_order is not None) and (len(win_pt_2_order) is 4):
            sorted_by_first = sorted(win_pt_2_order, key=lambda tup: tup[0])
            sorted_by_second = sorted(win_pt_2_order, key=lambda tup: tup[1])
            # print("First sorted ", sorted_by_first)
            # print("Second sorted ", sorted_by_second)
            sorted_clockwise = []
            if (sorted_by_first[0] is sorted_by_second[0]):
                sorted_clockwise.append(sorted_by_first[0])  # upper left
                sorted_clockwise.append(sorted_by_first[1])  # lower left
            elif (sorted_by_first[0] is sorted_by_second[1]):
                sorted_clockwise.append(sorted_by_first[0])
                sorted_clockwise.append(sorted_by_first[1])
            else:
                sorted_clockwise.append(sorted_by_first[1])
                sorted_clockwise.append(sorted_by_first[0])

            if (sorted_by_first[3] is sorted_by_second[3]):
                sorted_clockwise.append(sorted_by_first[3])  # lower right
                sorted_clockwise.append(sorted_by_first[2])  # upper right
            elif (sorted_by_first[3] is sorted_by_second[2]):
                sorted_clockwise.append(sorted_by_first[3])
                sorted_clockwise.append(sorted_by_first[2])
            else:
                sorted_clockwise.append(sorted_by_first[2])
                sorted_clockwise.append(sorted_by_first[3])

            # print("Ordered points are\n", sorted_clockwise)
            return sorted_clockwise

    def read_projection_matrix(self, projection_matrix_no):
        with open(self.projectionMatrices[projection_matrix_no], 'r') as f:
            projection = f.readlines()

        # print(self.projectionMatrices[projection_matrix_no])

        projection.remove(projection[0])
        projection = list(map(lambda a: a.split(), projection))
        projection = np.float64(projection)

        return projection

    def compute_3d_points_ground(self):
        """
        Take the 2d coordinates and 3D coordinates in multiple images
        and compute the relative scale, rotation and translation
        :return: scale, rotation_matrix, translation_vector
        """
        num_points = len(next(iter(self.select_pt_point.values())))
        A_mat_dict = {i:{'A': []} for i in range(num_points)}

        # A_mat_dict = {key: [] for key in self.select_pt_point.keys()}
        for key, val in self.select_pt_point.items():
            proj = self.read_projection_matrix(key)
            val = np.array(val)

            for i in range(val.shape[0]):
                cur_mat = np.array([val[i, 0] * proj[2, :] - proj[0, :], val[i, 1] * proj[2, :] - proj[1, :]])
                A_mat_dict[i]['A'].append(cur_mat)

        for key, val in A_mat_dict.items():
            A_mat_dict[key]['A'] = np.concatenate(val['A'], axis=0)
            A_mat_dict[key]['A_3'] = A_mat_dict[key]['A'][:,:3]
            A_mat_dict[key]['b'] = -1.0 * A_mat_dict[key]['A'][:, -1:]
            A_mat_dict[key]['sol'] = np.linalg.lstsq( A_mat_dict[key]['A_3'],  A_mat_dict[key]['b'], rcond=None)[0]
            # print(A_mat_dict[key]['sol'])

        points3D = np.concatenate([val['sol'] for val in A_mat_dict.values()], axis=1)
        print(points3D)

        return points3D


    def compute_3d_points_ground_opencv(self):
        """
        Take the 2d coordinates and projection matrices and
        compute the corresponding 3D points
        :return: points3D np array in shape 3xN
        """
        num_points = len(next(iter(self.select_pt_point.values())))
        A_mat_dict = {i:{'A': []} for i in range(num_points)}
        proj_list = []
        point_2d_list = []

        # A_mat_dict = {key: [] for key in self.select_pt_point.keys()}
        for key, val in self.select_pt_point.items():
            proj = np.float32(self.read_projection_matrix(key))
            val = np.array(val, dtype=np.float32)
            proj_list.append(proj)
            point_2d_list.append(val.T)

        # print(proj_list[0].shape, proj_list[1].shape, point_2d_list[0].shape, point_2d_list[1].shape)
        points4D = cv2.triangulatePoints(proj_list[0], proj_list[1], point_2d_list[0], point_2d_list[1])
        points3D = points4D[:3,:] / points4D[-1,:]
        print("Estimated 3D points from user selection:     ")
        print(points3D)

        return points3D

    def align_user_input(self, user_3D_points):

        # points3D_model = self.compute_3d_points_ground_opencv()
        points3D_model = self.compute_3d_points_ground()

        rot, trans, scale, trans_error = align(points3D_model, user_3D_points)
        print("Scale is {}".format(scale))
        estimated_points = scale * np.dot(rot, points3D_model) + trans
        print("Estimated points are ")
        print(estimated_points)
        estimated_transform = {"scale": scale, "rot": rot, "trans": trans}

        return estimated_transform

    def change_point_num(self):
        try:
            return len(next(iter(self.select_pt_point.values())))
        except:
            print("Error!! Points are not selected!!")
            return 4







