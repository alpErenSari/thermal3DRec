import numpy as np
import cv2
from plyProcess.plyfileColmap import PlyData, PlyElement

class plyProcess:
    def __init__(self, file_name):
        self.file_name = file_name
        self.epsilon = 50 # for having a boundary
        self.method = "colmap"
        self.properties_to_load = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']

    def change_method(self, method):
        assert (method is "colmap") or (method is "openMVG")
        self.method = method

    def set_file_name(self, file_name):
        self.file_name = file_name

    def read_ply_file_openmvg(self):
        with open(self.file_name, 'r') as f:
            self.ply = f.readlines()

        self.ply = [a.split() for a in self.ply]
        self.ply_ = np.float64(self.ply[13:]) # take the number part and transform into np array
        self.ply = self.ply[:13] # get rid of the number part to save some memory

        return self.ply_

    def read_ply_file_colmap(self):
        with open(self.file_name, 'r') as f:
            self.ply = f.readlines()

        self.ply = [a.split() for a in self.ply]
        self.ply_ = np.float64(self.ply[17:])[:,:9] # take the number part and transform into np array
        self.ply = self.ply[:17] # get rid of the number part to save some memory

        return self.ply_

    def read_ply_file_colmap_2(self):
        # with open(self.file_name, 'r') as f:
        self.plydata = PlyData.read(self.file_name)

        # properties_to_load = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        # available_properties = [prop.name for prop in self.plydata.elements[0].properties]
        self.ply_ = np.stack(
            [self.plydata.elements[0].data[prop] for prop in self.properties_to_load], axis=-1)

        return self.ply_

        # cur_file_name = '/media/eren/1.0 TB/eren/mimari-proje/class1/class44_14_02019/Class44_7_70_2_COLMAP/colmap_output/dense/0/fused.ply'
        # # with open(cur_file_name, 'r') as f:
        # plydata = PlyData.read(cur_file_name)

    def read_ply_file(self, method):
        # assert (method is "colmap") or (method is "openMVG")
        if method is 2:
            # use openmvg method is method is 2
            return self.read_ply_file_openmvg()
        else:
            return self.read_ply_file_colmap_2()


    def rgb2gray(self):

        r, g, b = self.colors[0,:], self.colors[1,:], self.colors[2,:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return np.dot(self.colors[:,0,:], np.array([0.299, 0.587, 0.114]).T)

    def add_points(self, addition_points):
        self.ply_ = np.concatenate((self.ply_, addition_points), axis=0)


    #this method takes the thermal image and the projection matrix
    # and it replaces bgr pixel intensities with thermal pixel intensities
    def add_thermal_points(self, project_the, img_thermal):
        self.img_thermal_rgb = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2RGB)

        self.coordinates = self.ply_[:, 0:4].T
        self.coordinates[3, :] = 1

        # project the points and normalize
        self.coordinates_ = np.dot(project_the, self.coordinates)
        self.coord_norm = self.coordinates_ / self.coordinates_[2, :]

        self.mask = (self.coord_norm[0, :] > 0) & (self.coord_norm[1, :] > 25) & \
                    (self.coord_norm[0, :] < self.img_thermal_rgb.shape[1]-20) &   \
                    (self.coord_norm[1, :] < self.img_thermal_rgb.shape[0]-30) & \
                    (self.coordinates_[2, :] > 0)
        self.items = np.argwhere(self.mask)

        self.coord_norm = np.int32(self.coord_norm)
        self.ply_[self.items, -3:] = self.img_thermal_rgb[self.coord_norm[1, self.items], \
                                self.coord_norm[0, self.items], :]

        return self.ply_

    def get_selected_coord_for_plane(self, img, project_the, rectangle_point):
        # self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # this method takes the selection area in one image
        # and finds the corresponding 3D points by using the projection matrix

        self.coordinates_rgb = self.ply_[:, 0:4].T
        self.coordinates_rgb[3, :] = 1

        # project the points and normalize
        self.coordinates_rgb_ = np.dot(project_the, self.coordinates_rgb)
        self.coord_norm_rgb = self.coordinates_rgb_ / self.coordinates_rgb_[2, :]

        # print("The projection: ", project_the)

        # find the projected points with
        # 4th line corresponds to x, 5th y
        self.mask_rgb = (self.coord_norm_rgb[0, :] > 0) & (self.coord_norm_rgb[1, :] > 0) & \
                    (self.coord_norm_rgb[0, :] < img.shape[1]) & \
                    (self.coord_norm_rgb[1, :] < img.shape[0]) & \
                    (self.coord_norm_rgb[0, :] > rectangle_point[0][0]) & \
                    (self.coord_norm_rgb[1, :] > rectangle_point[0][1]) & \
                    (self.coord_norm_rgb[0, :] < rectangle_point[1][0]) & \
                    (self.coord_norm_rgb[1, :] < rectangle_point[1][1]) & \
                    (self.coordinates_rgb_[2, :] > 0)
        self.items_rgb = np.argwhere(self.mask_rgb)

        self.coord_norm_rgb = np.int32(self.coord_norm_rgb)
        # now it's time for taking the 3D points on plane

        plane_coordinates = self.coordinates_rgb[:, self.items_rgb[:, 0]]
        mean_plane_coord = np.mean(plane_coordinates, axis=1, keepdims=True)
        # now fit a plane equation to these points
        self.colors = self.ply_[self.items_rgb, -3:]
        self.gray = self.rgb2gray()
        # self.mean_temp = (np.mean(self.gray)/255)*(35.0-15.0) + 15.0
        self.mean_temp = np.mean(self.gray)

        # self.ply_[self.items_rgb, -3:] = [0,255,0]

        return plane_coordinates, self.mean_temp

    def exclude_window_points(self, project_the, window_pt, img_shape):

        print(window_pt)
        window_pt = np.array(window_pt)
        print(window_pt)
        rectangle_point = np.zeros((2,2))
        rectangle_point[0][:] = np.min(window_pt, axis=0)
        rectangle_point[1][:] = np.max(window_pt, axis=0)
        # this method takes the selection area in one image
        # and finds the corresponding 3D points by using the projection matrix

        self.coordinates_rgb = self.ply_[:, 0:4].T
        self.coordinates_rgb[3, :] = 1

        # project the points and normalize
        self.coordinates_rgb_ = np.dot(project_the, self.coordinates_rgb)
        self.coord_norm_rgb = self.coordinates_rgb_ / self.coordinates_rgb_[2, :]

        # print("The projection: ", project_the)

        # find the projected points with
        # 4th line corresponds to x, 5th y
        self.mask_rgb = (self.coord_norm_rgb[0, :] > 0) & (self.coord_norm_rgb[1, :] > 0) & \
                    (self.coord_norm_rgb[0, :] < img_shape[1]) & \
                    (self.coord_norm_rgb[1, :] < img_shape[0]) & \
                    (self.coord_norm_rgb[0, :] > rectangle_point[0][1]) & \
                    (self.coord_norm_rgb[1, :] > rectangle_point[0][0]) & \
                    (self.coord_norm_rgb[0, :] < rectangle_point[1][1]) & \
                    (self.coord_norm_rgb[1, :] < rectangle_point[1][0]) & \
                    (self.coordinates_rgb_[2, :] > 0)
        # self.items_rgb = np.bitwise_not(np.argwhere(self.mask_rgb))
        self.items_rgb = np.argwhere(np.bitwise_not(self.mask_rgb))

        # self.coord_norm_rgb = np.int32(self.coord_norm_rgb)
        # now it's time for taking the 3D points on plane

        self.ply_ = self.ply_[self.items_rgb[:, 0], :]
        print("Rectangle points are ")
        print(rectangle_point)
        print("Items shape")
        print(self.items_rgb.shape)
        print("coord_norm_rgb shape")
        print(self.coord_norm_rgb.shape)
        # mean_plane_coord = np.mean(plane_coordinates, axis=1, keepdims=True)
        # now fit a plane equation to these points
        # self.mean_temp = (np.mean(self.gray)/255)*(35.0-15.0) + 15.0

        # self.ply_[self.items_rgb, -3:] = [0,255,0]

        # return self.ply_

    def get_selected_door_pt(self, project_the, door_pt):
        # self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # this method takes the selection area in one image
        # and finds the corresponding 3D points by using the projection matrix
        self.coordinates_rgb = self.ply_[:, 0:4].T
        self.coordinates_rgb[3, :] = 1

        # project the points and normalize
        self.coordinates_rgb_ = np.dot(project_the, self.coordinates_rgb)
        self.coord_norm_rgb = self.coordinates_rgb_ / self.coordinates_rgb_[2, :]
        self.coord_norm_rgb = self.coord_norm_rgb[:2,:]
        # print("The projection: ", project_the)

        # find the projected points with
        # smallest distance to the points

        #ensure that door_pt is list
        door_pt = list(door_pt)
        door_pt_locations = []
        for point in door_pt:
            diff = self.coord_norm_rgb - np.array(point).reshape((-1,1))
            diff_norm = np.linalg.norm(diff, axis=0)
            loc_min = np.argmin(diff_norm)
            door_pt_locations.append(loc_min)

        print("loc min is", door_pt_locations)
        return self.coordinates_rgb[:, door_pt_locations]

    def get_selected_window_pt(self, project_the, window_pt):
        # self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # this method takes the selection area in one image
        # and finds the corresponding 3D points by using the projection matrix
        self.coordinates_rgb = np.copy(self.ply_[:, 0:4].T)
        self.coordinates_rgb[3, :] = 1

        # project the points and normalize
        self.coordinates_rgb_ = np.dot(project_the, self.coordinates_rgb)
        self.coord_norm_rgb = self.coordinates_rgb_ / self.coordinates_rgb_[2, :]
        self.coord_norm_rgb = self.coord_norm_rgb[:2,:]
        # print("The projection: ", project_the)

        # find the projected points with
        # smallest distance to the points

        #ensure that window_pt is list
        window_pt = list(window_pt)
        window_pt_locations = []
        for point in window_pt:
            diff = self.coord_norm_rgb - np.array(point).reshape((-1,1))
            diff_norm = np.linalg.norm(diff, axis=0)
            loc_min = np.argmin(diff_norm)
            window_pt_locations.append(loc_min)

        print("loc min is", window_pt_locations)
        return self.coordinates_rgb[:, window_pt_locations]


    def save_point_cloud(self, name_to_save):
        self.ply[2][2] = str(self.ply_.shape[0])
        with open(name_to_save, 'w') as the_file:
            for item in self.ply[:13]:
                the_file.write(' '.join(item) + '\n')
            np.savetxt(the_file, self.ply_, delimiter=' ', fmt='%6f %6f %6f %6f %6f %6f %d %d %d')

    def save_point_cloud_lib(self, name_to_save):
        assert len(self.properties_to_load) == self.ply_.shape[1]
        self.properties_to_load
        vertex = np.copy(self.ply_).astype(dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                         ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        ele = []
        for i in range(len(self.properties_to_load)):
            ele.append(PlyElement.describe(vertex[:,0], self.properties_to_load[i]))

        PlyData(ele).write(name_to_save)

    def save_modified_point_cloud(self, point_cloud, name_to_save):
        self.ply[2][2] = str(point_cloud.shape[0])
        with open(name_to_save, 'w') as the_file:
            for item in self.ply[:13]:
                the_file.write(' '.join(item) + '\n')
            np.savetxt(the_file, point_cloud, delimiter=' ', fmt='%6f %6f %6f %6f %6f %6f %d %d %d')



