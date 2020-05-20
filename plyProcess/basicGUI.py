"""
This is the basic gui implementation
I hope it will work
by Alp Eren SARI
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from time import gmtime, strftime
from PIL import Image
import PIL.ImageTk
import glob, os
import pickle
import pandas as pd
from tqdm import tqdm
from plyProcess.thermalPly import thermal
import subprocess
from plyProcess.ransac import angle_between_planes, save_geometry_off, find_rotation_from_two_vectors, \
    vol_calculator_box, calculate_gt_vertex_error, save_obj, load_obj
from plyProcess.openMVGReconstruct import openMVGReconstruct
from plyProcess.colmapReconstructor import colmapReconstruct
from plyProcess.xmlOutput import xmlOutput
from plyProcess.rvalue import RCalculation

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.img_list = []
        self.img_window_list = []
        self.img_door_list = []
        self.folder_path = ''
        self.windowPoints = []
        self.doorPoints = []
        self.t_inside = tk.DoubleVar(value=25.0)
        self.t_outside = tk.DoubleVar(value=5.0)
        self.t_reflected = tk.DoubleVar(value=24.0)
        self.ransac_param = tk.DoubleVar(value=0.05)
        self.penalty_param = tk.DoubleVar(value=1.0)
        self.r_value = RCalculation(0, 21.1)
        self.rec_selection = tk.IntVar(value=1)
        self.rec_method_selection = tk.IntVar(value=1)
        self.camera_selection = tk.IntVar(value=1)
        self.room_selection = tk.IntVar(value=44)
        self.method_selection = tk.IntVar(value=1)
        self.program_mode = tk.IntVar(value=1)
        self.pack()
        self.is_images_loaded = False
        self.electroptic_images_loaded_100 = []
        self.df = pd.DataFrame(columns=['Volume', 'Vertex Error', 'Ortho Error'])
        self.baseline_points = np.load('/media/eren/1.0 TB/eren/mimari-proje/class1/' +
                                       'class44_14_02019/Class44_4_14062019_AprilTag/baseline.npy')
        self.image_extensions_to_load = ['*.jpg', '*.JPG', '*.png', '*.PNG']
        self.electroptic_images = []
        self.num_user_3D_point_input = 4
        self.user3D_point = np.array([[-1.16, -1.5, 0], [-0.47, 0, 1.20], [0, -1.90, 0.46], [-0.47, 0, 0.37]])
        self.variable_list_3d_pts = [[tk.DoubleVar(value=self.user3D_point[i,j]) for j in range(3)] for i in
                                     range(self.num_user_3D_point_input)]
        self.estimated_transform_found = False
        self.create_widgets()



    def create_widgets(self):
        # self.hi_there = tk.Button(self)
        # self.hi_there["text"] = "Hello World\n(click me)"
        # self.hi_there["command"] = self.say_hi
        # self.hi_there.pack(side="top")

        self.selection = tk.Radiobutton(self, variable=self.rec_selection, value=1)
        self.selection["text"] = "Only 3D Reconstruction"
        # self.selection["variable"] = self.rec_selection
        self.selection.pack()

        self.selection_2 = tk.Radiobutton(self, variable=self.rec_selection, value=2)
        self.selection_2["text"] = "3D Reconstruction with\n Thermal Data"
        # self.selection["variable"] = self.rec_selection
        self.selection_2.pack()

        self.selection_colmap = tk.Radiobutton(self, variable=self.rec_method_selection, value=1)
        self.selection_colmap["text"] = "3D Reconstruction with\n COLMAP"
        # self.selection["variable"] = self.rec_selection
        self.selection_colmap.pack()

        self.selection_openmvg = tk.Radiobutton(self, variable=self.rec_method_selection, value=2)
        self.selection_openmvg["text"] = "3D Reconstruction with\n OpenMVG"
        # self.selection["variable"] = self.rec_selection
        self.selection_openmvg.pack()

        self.selection_5 = tk.Radiobutton(self, variable=self.camera_selection, value=1)
        self.selection_5["text"] = "3D Reconstruction with\n Nikon D90"
        # self.selection["variable"] = self.rec_selection
        self.selection_5.pack()

        self.selection_3 = tk.Radiobutton(self, variable=self.camera_selection, value=2)
        self.selection_3["text"] = "3D Reconstruction with\n Koral Xiaomi"
        # self.selection["variable"] = self.rec_selection
        self.selection_3.pack()

        self.selection_4 = tk.Radiobutton(self, variable=self.camera_selection, value=3)
        self.selection_4["text"] = "3D Reconstruction with\n Thermal RGB Camera"
        # self.selection["variable"] = self.rec_selection
        self.selection_4.pack()

        self.selection_mode = tk.Radiobutton(self, variable=self.program_mode, value=1)
        self.selection_mode["text"] = "Normal Mode"
        # self.selection["variable"] = self.rec_selection
        self.selection_mode.pack()

        self.selection_mode_2 = tk.Radiobutton(self, variable=self.program_mode, value=2)
        self.selection_mode_2["text"] = "Test Mode"
        # self.selection["variable"] = self.rec_selection
        self.selection_mode_2.pack()

        self.selection_method = tk.Radiobutton(self, variable=self.method_selection, value=1)
        self.selection_method["text"] = "Baseline Method"
        # self.selection["variable"] = self.rec_selection
        self.selection_method.pack()

        self.selection_method_2 = tk.Radiobutton(self, variable=self.method_selection, value=2)
        self.selection_method_2["text"] = "Shoebox Method"
        # self.selection["variable"] = self.rec_selection
        self.selection_method_2.pack()



        self.selection_room = tk.Radiobutton(self, variable=self.room_selection, value=44)
        self.selection_room["text"] = "Room 44"
        # self.selection["variable"] = self.rec_selection
        self.selection_room.pack()

        self.selection_room_2 = tk.Radiobutton(self, variable=self.room_selection, value=45)
        self.selection_room_2["text"] = "Room 45"
        # self.selection["variable"] = self.rec_selection
        self.selection_room_2.pack()

        self.selection_room_3 = tk.Radiobutton(self, variable=self.room_selection, value=43)
        self.selection_room_3["text"] = "Room 43"
        # self.selection["variable"] = self.rec_selection
        self.selection_room_3.pack()

        tk.Label(self, text="Penalty Parameter").pack()
        self.get_penalty_param = tk.Entry(self, textvariable=self.penalty_param)
        self.get_penalty_param.pack()

        self.music = tk.Button(self)
        self.music["text"] = "Click to select\n desired folder"
        self.music["command"] = self.create_window_folder_selection   # browse_button
        self.music.pack()

        self.temp_but = tk.Button(self)
        self.temp_but["text"] = "Temperature"
        self.temp_but["fg"] = "blue"
        self.temp_but["command"] = self.create_window_get_temp
        self.temp_but.pack()

        self.point_but = tk.Button(self)
        self.point_but["text"] = "3D Points User Input"
        self.point_but["fg"] = "green"
        self.point_but["command"] = self.create_window_get_3D_points
        self.point_but.pack()


        self.rec = tk.Button(self)
        self.rec["text"] = "3D Reconstruction"
        self.rec["fg"] = "red"
        self.rec["command"] = self.reconstruct
        self.rec.pack()

        self.pictures_point_select = tk.Button(self)
        self.pictures_point_select["text"] = "Pictures for \n 3D Point Selection"
        self.pictures_point_select["command"] = self.create_window_4_point
        self.pictures_point_select.pack()

        self.pictures = tk.Button(self)
        self.pictures["text"] = "Pictures for \nPlane Selection"
        self.pictures["command"] = self.create_window
        self.pictures.pack()

        self.pictures_window = tk.Button(self)
        self.pictures_window["text"] = "Pictures for \n Window Selection"
        self.pictures_window["command"] = self.create_window_4_window
        self.pictures_window.pack()

        self.pictures_door = tk.Button(self)
        self.pictures_door["text"] = "Pictures for \n Door Selection"
        self.pictures_door["command"] = self.create_window_4_door
        self.pictures_door.pack()

        self.plane_point = tk.Button(self)
        self.plane_point["text"] = "Point Selecter"
        self.plane_point["command"] = self.point_selector
        self.plane_point.pack()

        self.plane_window = tk.Button(self)
        self.plane_window["text"] = "Window Selecter"
        self.plane_window["command"] = self.window_selector
        self.plane_window.pack()

        self.plane_door = tk.Button(self)
        self.plane_door["text"] = "Door Selecter"
        self.plane_door["command"] = self.door_selector
        self.plane_door.pack()

        self.plane = tk.Button(self)
        self.plane["text"] = "Plane selecter"
        self.plane["command"] = self.plane_selecter
        self.plane.pack()

        self.run_process = tk.Button(self)
        self.run_process["text"] = "RUN"
        self.run_process["command"] = self.geometry_calculator
        self.run_process["fg"] = "red"
        self.run_process.pack()

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.quit)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")

    def plane_selecter(self):
        # print("Image list: ", self.img_list)
        if(not hasattr(self, 'th')):
            self.th = thermal(self.param)

        if (self.rec_selection.get() is 1):
            self.ply_ = self.th.only_read_ply(self.rec_method_selection.get())
        elif (self.rec_selection.get() is 2):
            self.ply_ = self.th.add_thermal_points_all(self.rec_method_selection.get())
            self.th.save_processed_point_cloud("magical_thermal_points")

        # self.normals, self.errors, self.mean_temps, self.homografies = self.th.thermal_image_point_select(box_method=self.method_selection.get(),
        #                                                                                                   ransac_param=self.ransac_param,
        #                                                                                                   plane_list=self.img_list,
        #                                                                                 method=self.rec_selection.get(),
        #                                                                                 mode=1)

        self.th.thermal_image_point_select(box_method=self.method_selection.get(),
                                           ransac_param=self.ransac_param.get(),
                                           plane_list=self.img_list,
                                           method=self.rec_selection.get(),
                                           mode=1)



        # number_of_runs = 1
        # if self.program_mode.get() is 2:
        #     number_of_runs = 100
        #
        #
        # for i in range(number_of_runs):
        #     self.normals, self.errors, self.mean_temps, self.ortho_error = self.th.thermal_rectangle_prism_fitting(
        #         self.method_selection.get(),
        #         self.ransac_param, plane_list=self.img_list,
        #         method=self.rec_selection.get()
        #     )
        #
        #     my_rotation = find_rotation_from_two_vectors(self.normals[4][:3,0], np.array([0, 0, 1]))
        #     # my_rotation2 = find_rotation_from_two_vectors(self.normals[0][:3, 0], np.array([0, 0, 1]))
        #
        #
        #     # self.normals[5] = np.array([[0],[0],[1],[0]])
        #     # self.normals[3], _ = estimate(self.windowPoints[0].T)
        #     other_normals = self.normal_reader("/media/eren/1.0 TB/eren/mimari-proje/class1/nikon_17_04_2019/computed_normals.pickle")
        #     other_normals_2 = self.normal_reader(
        #         "/media/eren/1.0 TB/eren/mimari-proje/class2/fullRoom/computed_normals.pickle")
        #     # self.normals[1] = other_normals_2[3]
        #     # # self.normals[1] = other_normals[3]
        #     #for class 43
        #     if(self.room_selection.get() is 43):
        #         self.normals[0] = other_normals[0]
        #         self.normals[2] = other_normals[2]
        #         self.normals[3] = other_normals[1]
        #         self.normals[4] = other_normals[4]
        #         self.normals[5] = other_normals[5]
        #     elif(self.room_selection.get() is 45):
        #         # #for class 45
        #         self.normals[1] = other_normals[3]
        #         self.normals[2] = other_normals[2]
        #         self.normals[4] = other_normals[4]
        #         self.normals[5] = other_normals[5]
        #
        #     normalized_normals = [-x / x[2] for x in self.normals]
        #     print(normalized_normals)
        #     print("Error for rectangle prism is \n", sum(self.errors))
        #     # print("Mean temperetures are\n", self.mean_temps)
        #     # for i in range(len(self.normals)-1):
        #     #     print("The angle between planes", angle_between_planes(self.normals[i], self.normals[i+1])/3.1415926)
        #
        #     if (self.room_selection.get() is 44):
        #         self.windowPoints = self.th.find_3_pt_from_pixel_and_projection(self.normals,
        #                                                                         img_window_list=self.img_window_list,
        #                                                                         normal_no=[2, 3])
        #     else:
        #         self.windowPoints = self.th.find_3_pt_from_pixel_and_projection(self.normals,
        #                                                                         img_window_list=self.img_window_list, normal_no=[2])
        #     if(self.room_selection.get() is 44):
        #         self.doorPoints = self.th.find_3_pt_from_pixel_and_projection(self.normals,
        #                                                                       img_window_list=self.img_door_list, normal_no=[3], mode="door")
        #     else:
        #         self.doorPoints = self.th.find_3_pt_from_pixel_and_projection(self.normals,
        #                                                                       img_window_list=self.img_door_list,
        #                                                                       normal_no=[0], mode="door")
        #     # set door 3 for class1, for class 2 it is 0
        #     # print("windows points shape is ", self.windowPoints[0])
        #     self.corners = self.th.compute_volume_corners(self.normals)
        #     origin_corner = self.corners[1]
        #     self.corners = [corner - origin_corner for corner in self.corners]
        #     self.corners = [np.dot(my_rotation, corner) for corner in self.corners]
        #     for corners in self.windowPoints:
        #         corners[:3, :] = corners[:3, :] - np.reshape(origin_corner, (-1, 1))
        #         corners[:3,:] = np.dot(my_rotation, corners[:3, :])
        #     for corners in self.doorPoints:
        #         corners[:3, :] = corners[:3, :] - np.reshape(origin_corner, (-1, 1))
        #         corners[:3, :] = np.dot(my_rotation, corners[:3, :])
        #     # self.windowPoints = [np.dot(my_rotation, corners[:3, :]) for corners in self.windowPoints]
        #     # self.doorPoints = [np.dot(my_rotation, corners[:3, :]) for corners in self.doorPoints]
        #     print("Corners are ", self.corners)
        #     # inverse_corner = self.corners[2]
        #     # beginning_corner = self.corners[7]
        #     # volume_rect = abs(inverse_corner[0] - beginning_corner[0]) * \
        #     #               abs(inverse_corner[1] - beginning_corner[1]) * abs(inverse_corner[2] - beginning_corner[2])
        #
        #     volume_rect = vol_calculator_box(self.corners)
        #
        #     corners_ = np.array(self.corners)
        #     error_wt_gt = np.mean(np.linalg.norm(corners_ - self.baseline_points, axis=1))
        #
        #     self.df = self.df.append(
        #         pd.DataFrame({'Volume': [volume_rect], 'Vertex Error': [error_wt_gt], 'Ortho Error': [self.ortho_error]})
        #     )
        #     if number_of_runs - i is 1:
        #         print("Volume of room is ", volume_rect)
        #         print("Corners distance error is ", error_wt_gt)
        #         print("Orthonogality error is ", self.ortho_error)
        #
        # # print(corners_)
        # # print(self.baseline_points)
        # if(self.rec_selection.get() is 1):
        #     self.r_values_calculated = [0 for i in range(len(self.normals))]
        # elif(self.rec_selection.get() is 2):
        #     self.r_value = RCalculation(self.t_inside, self.t_outside)
        #     self.r_values_calculated = self.r_value.calculate_R_value_list(self.mean_temps, self.t_reflected)
        # print("The R values is ", self.r_values_calculated)
        # # self.normal_saver(self.normals)
        # self.xml_saver(self.corners)
        # self.th.save_processed_point_cloud("magical_thermal_points")
        #
        # csv_name = 'my_error_outs_baseline.csv'
        # if self.method_selection.get() is 2:
        #     csv_name = 'my_error_outs_shoebox.csv'
        # self.df.to_csv(csv_name)

    def geometry_calculator(self):
        df = pd.DataFrame(columns=['Volume', 'Vertex Error', 'Ortho Error', 'Angle Error N', 'Angle Error E',
                                   'Angle Error S', 'Angle Error W', 'Angle Error C', 'Angle Error F'])
        run_path = os.path.join(self.folder_path, "runs")
        if not os.path.exists(run_path):
            os.mkdir(run_path)

        number_of_runs = 1
        self.th.change_penalty_param(self.penalty_param.get())
        if self.program_mode.get() is 2:
            number_of_runs = 100

        for i in range(number_of_runs):
            self.normals, self.errors, self.mean_temps, self.ortho_error = self.th.thermal_rectangle_prism_fitting(
                self.method_selection.get(),
                self.ransac_param.get(), plane_list=self.img_list,
                method=self.rec_selection.get()
            )

            my_rotation1 = find_rotation_from_two_vectors(self.normals[4][:3, 0], np.array([0, 0, 1]))
            my_rotation2 = find_rotation_from_two_vectors(self.normals[0][:3, 0], np.array([1, 0, 0]))
            my_rotation = np.dot(my_rotation1, my_rotation2)

            # self.normals[5] = np.array([[0],[0],[1],[0]])
            # self.normals[3], _ = estimate(self.windowPoints[0].T)
            other_normals = self.normal_reader(
                "/media/eren/1.0 TB/eren/mimari-proje/class1/nikon_17_04_2019/computed_normals.pickle")
            other_normals_2 = self.normal_reader(
                "/media/eren/1.0 TB/eren/mimari-proje/class2/fullRoom/computed_normals.pickle")
            # self.normals[1] = other_normals_2[3]
            # # self.normals[1] = other_normals[3]
            # for class 43
            if (self.room_selection.get() is 43):
                self.normals[0] = other_normals[0]
                self.normals[2] = other_normals[2]
                self.normals[3] = other_normals[1]
                self.normals[4] = other_normals[4]
                self.normals[5] = other_normals[5]
            elif (self.room_selection.get() is 45):
                # #for class 45
                self.normals[1] = other_normals[3]
                self.normals[2] = other_normals[2]
                self.normals[4] = other_normals[4]
                self.normals[5] = other_normals[5]

            normalized_normals = [-x / x[2] for x in self.normals]
            print(normalized_normals)
            print("Error for rectangle prism is \n", sum(self.errors))
            # print("Mean temperetures are\n", self.mean_temps)
            # for i in range(len(self.normals)-1):
            #     print("The angle between planes", angle_between_planes(self.normals[i], self.normals[i+1])/3.1415926)

            if (self.room_selection.get() is 44):
                self.windowPoints = self.th.find_3_pt_from_pixel_and_projection(self.normals,
                                                                                img_window_list=self.img_window_list,
                                                                                normal_no=[2, 3])
            else:
                self.windowPoints = self.th.find_3_pt_from_pixel_and_projection(self.normals,
                                                                                img_window_list=self.img_window_list,
                                                                                normal_no=[2])
            if (self.room_selection.get() is 44):
                self.doorPoints = self.th.find_3_pt_from_pixel_and_projection(self.normals,
                                                                              img_window_list=self.img_door_list,
                                                                              normal_no=[3], mode="door")
            else:
                self.doorPoints = self.th.find_3_pt_from_pixel_and_projection(self.normals,
                                                                              img_window_list=self.img_door_list,
                                                                              normal_no=[0], mode="door")
            # set door 3 for class1, for class 2 it is 0
            # print("windows points shape is ", self.windowPoints[0])

            self.corners = self.th.compute_volume_corners(self.normals)
            origin_corner = self.corners[1]
            self.corners = [corner - origin_corner for corner in self.corners]
            self.corners = [self.estimated_transform["scale"] * np.dot(my_rotation, corner) for corner in self.corners]
            for corners in self.windowPoints:
                corners[:3, :] = corners[:3, :] - np.reshape(origin_corner, (-1, 1))
                corners[:3, :] = np.dot(my_rotation, corners[:3, :])
            for corners in self.doorPoints:
                corners[:3, :] = corners[:3, :] - np.reshape(origin_corner, (-1, 1))
                corners[:3, :] = np.dot(my_rotation, corners[:3, :])

            # self.windowPoints = [np.dot(my_rotation, corners[:3, :]) for corners in self.windowPoints]
            # self.doorPoints = [np.dot(my_rotation, corners[:3, :]) for corners in self.doorPoints]
            print("Corners are ", self.corners)
            # inverse_corner = self.corners[2]
            # beginning_corner = self.corners[7]
            # volume_rect = abs(inverse_corner[0] - beginning_corner[0]) * \
            #               abs(inverse_corner[1] - beginning_corner[1]) * abs(inverse_corner[2] - beginning_corner[2])

            volume_rect = vol_calculator_box(self.corners)

            corners_ = np.array(self.corners)
            # error_wt_gt = np.mean(np.linalg.norm(corners_ - self.baseline_points, axis=1))
            error_wt_gt = calculate_gt_vertex_error(corners_, self.baseline_points)

            normal_orientation = [np.dot(my_rotation, normal_[:3,:]) for normal_ in self.normals]
            normal_orientation = [normal_ for normal_ in normal_orientation]
            normal_gt = [np.array([0,1,0]), np.array([1,0,0]), np.array([0,1,0]), np.array([1,0,0]), np.array([0,0,1]),
                         np.array([0,0,1])]
            normal_orientation_degree = [angle_between_planes(n1, n2)*180/np.pi for n1,n2 in
                                         zip(normal_orientation, normal_gt)]

            # df = df.append(
            #     pd.DataFrame(
            #         {'Volume': [volume_rect], 'Vertex Error': [error_wt_gt], 'Ortho Error': [self.ortho_error],
            #          'Angle Error N': [normal_orientation_degree[0][0]], 'Angle Error E': [normal_orientation_degree[1][0]],
            #         'Angle Error S': [normal_orientation_degree[2][0]], 'Angle Error W': [normal_orientation_degree[3][0]],
            #          'Angle Error C': [normal_orientation_degree[4][0]], 'Angle Error F':[normal_orientation_degree[5][0]]
            # }))

            df = df.append(
                pd.DataFrame(
                    {'Volume': [volume_rect],
                     'Vertex Error': [error_wt_gt]
                     })
            )
            if number_of_runs - i is 1:
                print("Volume of room is ", volume_rect)
                print("Corners distance error is ", error_wt_gt)
                print("Orthonogality error is ", self.ortho_error)
                print("Degree errors are ", normal_orientation_degree)

        # print(corners_)
        # print(self.baseline_points)
        if (self.rec_selection.get() is 1):
            self.r_values_calculated = [0 for i in range(len(self.normals))]
        elif (self.rec_selection.get() is 2):
            self.r_value = RCalculation(self.t_inside.get(), self.t_outside.get())
            self.r_values_calculated = self.r_value.calculate_R_value_list(self.mean_temps, self.t_reflected.get())
            self.th.save_processed_point_cloud("magical_thermal_points")
        print("The R values are ", self.r_values_calculated)
        # self.normal_saver(self.normals)
        self.xml_saver(self.corners)


        # df = df.append(
        #     pd.DataFrame(
        #         {'Volume': [0, ], 'Vertex Error': [0], 'Ortho Error': [0]})
        # )
        df = df.append(df.agg(['mean', 'std']))

        csv_name = 'my_error_outs_baseline_' + strftime("%Y-%m-%d-%H:%M:%S", gmtime())
        if self.method_selection.get() is 2:
            csv_name = 'my_error_outs_shoebox_' + strftime("%Y-%m-%d-%H:%M:%S", gmtime())


        csv_name = csv_name + '.csv'
        csv_path = os.path.join(run_path, csv_name)
        df.to_csv(csv_path)


    def xml_saver(self, corners):
        myxml = xmlOutput()
        types = ['Wall', 'Wall','Wall', 'Wall', 'Floor', 'Ceiling']
        directions = ['East', 'South', 'West', 'North', 'Up', 'Down']
        surface_pts = np.array([[3, 4, 5], [11, 12, 17], [31, 35, 38], [42, 49, 51]], dtype=np.float32)
        numberOfPoints = [4, 4, 4, 4, 4, 4]
        # numberOfWindows = self.th.assign_wall_to_window(self.windowPoints, 6)
        if (self.room_selection.get() is 44):
            numberOfWindows = [0, 1, 1, 0, 0, 0]
        else:
            numberOfWindows = [0, 1, 0, 0, 0, 0]
        if(self.room_selection.get() is 44):
            numberOfDoors = [0, 0, 1, 0, 0, 0]
        else:
            numberOfDoors = [0, 0, 0, 1, 0, 0] # for class 43

        surfacePoints = []
        # this is for the 6 surface rectangle prism case
        for i in range(4):
            srf_pts = [corners[(2*i)%8], corners[(2*i+1)%8], corners[(2*i+3)%8], corners[(2*i+2)%8]]
            surfacePoints.append(srf_pts)
        for i in range(2):
            srf_pts = [corners[i], corners[i+2], corners[i+4], corners[i+6]]
            surfacePoints.append(srf_pts)

        param = dict({'surfaceNumber': 6, 'surfacePoints': surfacePoints, 'surfaceTypes': types,
                      'numberOfPoints': numberOfPoints, 'numberOfWindows': numberOfWindows,
                       'windowPoints': self.windowPoints,'numberOfDoors': numberOfDoors,
                      'doorPoints':self.doorPoints, 'directions': directions,
                      'rValues': self.r_values_calculated})
        myxml.print_out('my_xml_file', param)
        save_geometry_off(corners, self.windowPoints, self.doorPoints, 'my_geometry.off')

    def normal_saver(self, normals):
        filename = os.path.join(self.root, "computed_normals.pickle")
        print("normals saved to ", filename)
        with open(filename, 'wb') as f:
            pickle.dump(normals, f)

    def normal_reader(self, filename):
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def window_selector(self):
        self.th.change_electrooptic_images(self.electroptic_images)
        self.th.window_selection_from_points(img_window_list=self.img_window_list)
        self.windowPoints = self.th.find_3_pt_from_pixel(img_window_list=self.img_window_list)

    def door_selector(self):
        self.th.change_electrooptic_images(self.electroptic_images)
        self.th.window_selection_from_points(img_window_list=self.img_door_list, mode="door")
        self.doorPoints = (self.th.find_3_pt_from_pixel(img_window_list=self.img_door_list))

    def point_selector(self):
        self.th.change_electrooptic_images(self.electroptic_images)
        self.th.window_selection_from_points(img_window_list=self.img_point_list, mode="point")
        self.num_user_3D_point_input = self.th.change_point_num()
        # self.th.compute_3d_points_ground_opencv()
        cur_3D_user_input = [[self.variable_list_3d_pts[i][j].get() for j in range(3)] for i in range(self.num_user_3D_point_input)]
        cur_3D_user_input = np.array(cur_3D_user_input).T
        self.estimated_transform = self.th.align_user_input(cur_3D_user_input)
        save_obj(self.estimated_transform, "estimated_transform", self.folder_path)


    def create_window_get_temp(self):
        # this method creates a new window for getting inside and outside temperatures
        window = tk.Toplevel(self.master)
        window.title("Temperature Input Window")
        tk.Label(window, text="Inside Temperature").grid(row=0)
        tk.Label(window, text="Outside Temperature").grid(row=1)
        tk.Label(window, text="Reflected Temperature").grid(row=2)
        tk.Label(window, text="RANSAC Parameter").grid(row=3)
        # tk.Label(window, text="Penalty Parameter").grid(row=4)
        get_temp = tk.Entry(window, textvariable=self.t_inside)
        # get_temp.insert(0, "0")
        get_temp.grid(row=0, column=1)
        get_temp_out = tk.Entry(window, textvariable=self.t_outside)
        # get_temp_out.insert(0, "0")
        get_temp_out.grid(row=1, column=1)
        get_temp_ref = tk.Entry(window, textvariable=self.t_reflected)
        # get_temp_ref.insert(0, "0")
        get_temp_ref.grid(row=2, column=1)
        get_ransac_param = tk.Entry(window, textvariable=self.ransac_param)
        # get_ransac_param.insert(0, "0")
        get_ransac_param.grid(row=3, column=1)
        # get_penalty_param = tk.Entry(window, textvariable=self.penalty_param)
        # get_penalty_param.insert(0, "0")
        # get_penalty_param.grid(row=4, column=1)
        quit_img = tk.Button(window, text="QUIT", fg="red",
                             command=window.destroy)
        quit_img.grid(row=5, columns=1)
        show_but = tk.Button(window, text="Show",
                             command=self.show_temp)
        show_but.grid(row=5, columns=2)

    def show_temp(self):
        print("The inside temperature is ", self.t_inside.get())
        print("The outside temperature is ", self.t_outside.get())
        print("the penaty parameter is ", self.penalty_param.get())

    def create_window_get_3D_points(self):
        # this method creates a new window for getting inside and outside temperatures
        window = tk.Toplevel(self.master)
        window.title("3D Points Input Window")
        for i,axis in enumerate(['X', 'Y', 'Z']):
            tk.Label(window, text="{} Axis".format(axis)).grid(row=0, column=i+1)

        for i in range(self.num_user_3D_point_input):
            tk.Label(window, text="Point Number {}".format(i)).grid(row=i+1, column=0)


        # tk.Label(window, text="Outside Temperature").grid(row=1)
        # tk.Label(window, text="Reflected Temperature").grid(row=2)
        # tk.Label(window, text="RANSAC Parameter").grid(row=3)
        # tk.Label(window, text="Penalty Parameter").grid(row=4)
        if self.num_user_3D_point_input != 4:
            self.variable_list_3d_pts = [[tk.DoubleVar(value=0.0) for j in range(3)] for i in
                                         range(self.num_user_3D_point_input)]

        entry_list = []
        for i in range(self.num_user_3D_point_input):
            for j in range(3):
                entry_list.append(tk.Entry(window, textvariable=self.variable_list_3d_pts[i][j]))
                entry_list[-1].grid(row=i+1, column=j+1)

        # get_temp = tk.Entry(window, textvariable=self.t_inside)
        # get_temp.grid(row=0, column=1)
        # get_temp_out = tk.Entry(window, textvariable=self.t_outside)

        quit_img = tk.Button(window, text="QUIT", fg="red",
                             command=window.destroy)
        quit_img.grid(row=5, columns=1)


    def create_window(self):
        window = ScrolledFrame(self.master)
        bbs = []
        # self.img_list = []
        # print("img_list before \n", self.img_list)
        self.img_list = []
        if not self.is_images_loaded:
            self.electroptic_images_loaded_100 = [cv2.resize(cv2.imread(img_path), (100,100)) for img_path in
                                                  self.electroptic_images]
            self.is_images_loaded = True

        for i, img_ in tqdm(enumerate(self.electroptic_images_loaded_100)):
            bb1 = Buttons3(window.inner, img_, num=i, root=self.folder_path, list_img=self.img_list)
            bbs.append(bb1)

        quit_img = tk.Button(window, text="QUIT", fg="red",
                              command=window.destroy)
        quit_img.grid(row=1, column=0)
        print("img_list after \n", self.img_list)

    def create_window_folder_selection(self):
        # this method is for main, image and thermal image folders
        print(self.rec_selection.get())
        window = tk.Toplevel(self.master)
        window.title("Folder Selection")
        bb1 = tk.Button(window, text="Main\n Folder Selection", command=self.browse_button)
        bb1.pack()
        bb2 = tk.Button(window, text="Image\n Folder Selection", command=self.browse_button_image)
        bb2.pack()
        if(self.rec_selection.get() is 2):
            bb3 = tk.Button(window, text="Thermal Image\n Folder Selection", command=self.browse_button_thermal)
            bb3.pack()
        quit_img = tk.Button(window, text="QUIT", fg="red",
                             command=window.destroy)
        quit_img.pack()


    def create_window_4_window(self):
        # this method creates a new window with image buttons
        # self.master.withdraw()
        self.img_window_list = []
        # window_w = tk.Toplevel()
        window_w = ScrolledFrame(self.master)
        window_w.title("Window Image Selection")
        bbs_w = []
        # self.img_list = []
        print(self.img_list)

        if not self.is_images_loaded:
            self.electroptic_images_loaded_100 = [cv2.resize(cv2.imread(img_path), (100,100)) for img_path in
                                                  self.electroptic_images]
            self.is_images_loaded = True

        for i, img_ in enumerate(self.electroptic_images_loaded_100):
            bb1 = Buttons3(window_w.inner, img_, num=i, root=self.folder_path, list_img=self.img_window_list)
            bbs_w.append(bb1)

        quit_img_w = tk.Button(window_w, text="QUIT", fg="red",
                              command=window_w.destroy)
        quit_img_w.grid(row=1, column=0)
        print("Window images are ", self.img_window_list)

    def create_window_4_door(self):
        # this method creates a new window with image buttons
        # self.master.withdraw()
        self.img_door_list = []
        window_d = ScrolledFrame(self.master)
        window_d.title("Door Image Selection")
        bbs_d = []

        if not self.is_images_loaded:
            self.electroptic_images_loaded_100 = [cv2.resize(cv2.imread(img_path), (100, 100)) for img_path in
                                                  self.electroptic_images]
            self.is_images_loaded = True

        for i, img_ in enumerate(self.electroptic_images_loaded_100):
            bb1 = Buttons3(window_d.inner, img_, num=i, root=self.folder_path, list_img=self.img_door_list)
            bbs_d.append(bb1)

        quit_img_d = tk.Button(window_d, text="QUIT", fg="red",
                              command=window_d.destroy)
        quit_img_d.grid(row=1, column=0)
        print("Door images are ", self.img_door_list)

    def create_window_4_point(self):
        # this method creates a new window with image buttons
        # self.master.withdraw()
        self.img_point_list = []
        window_d = ScrolledFrame(self.master)
        window_d.title("Point Image Selection")
        bbs_d = []

        if not self.is_images_loaded:
            self.electroptic_images_loaded_100 = [cv2.resize(cv2.imread(img_path), (100, 100)) for img_path in
                                                  self.electroptic_images]
            self.is_images_loaded = True

        for i, img_ in enumerate(self.electroptic_images_loaded_100):
            bb1 = Buttons3(window_d.inner, img_, num=i, root=self.folder_path, list_img=self.img_point_list)
            bbs_d.append(bb1)

        quit_img_d = tk.Button(window_d, text="QUIT", fg="red",
                              command=window_d.destroy)
        quit_img_d.grid(row=1, column=0)
        print("Point images are ", self.img_point_list)

    def music_open(self):
        music_open = subprocess.Popen(["firefox", "https://www.youtube.com/watch?v=Xkl36pu8aLY&list=RDUGILR4txS7w&index=8"])
        music_open.wait()

    def browse_button(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.folder_path = filename
        self.root = self.folder_path
        self.electroptic_images_path = os.path.join(self.folder_path, "outside_out_pmvs/reconstruction_sequential/PMVS/visualize/")
        print("folder path is ", self.folder_path)
        # print('image folder is {}'.format(self.electroptic_images_path))
        self.param = {
            'root': self.folder_path,
            'thermal_root': os.path.join(self.folder_path, ""),
            'rgb_intrinsic': os.path.join(self.folder_path, "parameters/electroptic_camera.txt"),
            'thermal_intrinsic': os.path.join(self.folder_path, "parameters/thermal_camera.txt"),
            'rot_diff_mat': os.path.join(self.folder_path, "parameters/rotation_difference.txt"),
            'trans_diff_': os.path.join(self.folder_path, "parameters/translation_difference_2.txt"),
            'begin_it': 43,
            'iterationNo': 46,
            'ransac_threshold': 0.05,
            'gui_image_number': 36,
            'box_method': self.method_selection.get()
        }
        if os.path.exists(os.path.join(self.folder_path, "estimated_transform.pkl")):
            self.estimated_transform = load_obj("estimated_transform", self.folder_path)
            try:
                print("A valid transformation file found, No need to point selection")
                print("Loaded scale is {}".format(self.estimated_transform["scale"]))
                self.estimated_transform_found = True
            except:
                print("Wrong estimated transform file!!!!")
                del self.estimated_transform

    def browse_button_image(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.image_path = filename
        pmvs_path = os.path.join(self.root, 'outside_out_pmvs/reconstruction_sequential/PMVS')
        print(filename)
        for ext in self.image_extensions_to_load:
            self.electroptic_images.extend(glob.glob(os.path.join(self.electroptic_images_path, ext)))
        self.electroptic_images.sort(reverse=False)
        print('{} images found in the images folder'.format(len(self.electroptic_images)))
        # print('Folder is {}'.format(self.electroptic_images_path))
        self.param['image_root'] = self.image_path
        if (os.path.exists(pmvs_path)):
            print('pmvs path exists')
            self.th = thermal(self.param)
            if (self.rec_selection.get() is 1):
                self.ply_ = self.th.only_read_ply(self.rec_method_selection.get())


    def browse_button_thermal(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.thermal_path = filename
        print(filename)
        self.param['thermal_root'] = self.thermal_path
        print("root is ", self.root)
        pmvs_path = os.path.join(self.root, 'outside_out_pmvs/reconstruction_sequential/PMVS')
        print("pmvs path is ", pmvs_path)
        if(os.path.exists(pmvs_path)):
            print('pmvs path exists')
            self.th = thermal(self.param)
            if (self.rec_selection.get() is 2):
                self.ply_ = self.th.add_thermal_points_all(self.rec_method_selection.get())
                self.th.save_processed_point_cloud("magical_thermal_points")
            else:
                raise ("Invalid operation mode!")


    def reconstruct(self):
        if (self.rec_selection.get() is 1):
            isThermal = self.camera_selection.get()
        else:
            isThermal = 3
        if(self.rec_method_selection.get() is 1):
            colmapRec = colmapReconstruct(self.folder_path, self.image_path)
            # colmapRec.reconstruct()
        else:
            mvgRec = openMVGReconstruct(self.folder_path, self.image_path, isThermal=isThermal)
            mvgRec.reconstruct()
        self.th = thermal(self.param)
        if(self.rec_selection.get() is 1):
            self.ply_ = self.th.only_read_ply(self.rec_method_selection.get())
        elif(self.rec_selection.get() is 2):
            self.ply_ = self.th.add_thermal_points_all(self.rec_method_selection.get())
            self.th.save_processed_point_cloud("magical_thermal_points")


class Buttons1(tk.Frame):
    # this is a button class which shows the image on the buttons
    # therefore, we can select the images to plane fit
    def __init__(self, master, electroptic_images, num=0, root='', list_img=[]):
        super().__init__(master)
        self.master = master
        self.count = 0
        self.num = num
        self.list_img = list_img
        # list the images
        # self.electroptic_images = glob.glob(
        #     root + "/outside_out_pmvs/reconstruction_sequential/PMVS/visualize/*.jpg")
        # self.electroptic_images.sort(reverse=False)
        self.electroptic_images = electroptic_images
        self.b3 = tk.Button(self.master, text="Pictures\n(click here)", command=self.select_image)
        # read the image and put it onto buttons
        self.photo = PIL.ImageTk.PhotoImage(Image.open(self.electroptic_images[self.num]).resize((100,100)))
        self.b3.config(image=self.photo, width="100", height="100")
        # place the buttons
        self.b3.grid(row=int(self.num/7)%7, column=self.num%7)
        # self.b3.pack()
        # self.pack()
        self.transform_matrix = (
            0, 0.157580, 0, 0,
            0, 0.715160, 0, 0,
            0, 0.119193, 0, 0 )

    def select_image(self):
        print("You selected the image {} !!".format(self.num))
        self.list_img.append(self.num)
        img = Image.open(self.electroptic_images[self.num]).resize((100, 100))
        self.photo = PIL.ImageTk.PhotoImage(img.convert("RGB", self.transform_matrix))
        self.b3.config(image=self.photo, width="100", height="100")

class Buttons2(tk.Frame):
    # this is a button class which shows the image on the buttons
    # therefore, we can select the images to plane fit
    def __init__(self, master, img_path, num=0, root='', list_img=[]):
        super().__init__(master)
        self.master = master
        self.count = 0
        self.num = num
        self.list_img = list_img
        # list the images
        self.electroptic_images = glob.glob(
            root + "/outside_out_pmvs/reconstruction_sequential/PMVS/visualize/*.jpg")
        self.electroptic_images.sort(reverse=False)
        self.b3 = tk.Button(self.master, text="Pictures\n(click here)", command=self.select_image)
        # read the image and put it onto buttons
        self.photo_np = cv2.resize(cv2.imread(img_path), (100,100))
        self.photo_np = cv2.cvtColor(self.photo_np, cv2.COLOR_BGR2RGB)
        # self.photo = PIL.ImageTk.PhotoImage(Image.open(img_path).resize((100,100)))
        self.photo = PIL.ImageTk.PhotoImage(Image.fromarray(self.photo_np))
        self.b3.config(image=self.photo, width="100", height="100")
        # place the buttons
        self.b3.grid(row=int(self.num/5), column=self.num%5)
        # self.b3.pack()
        # self.pack()
        self.transform_matrix = (
            0, 0.157580, 0, 0,
            0, 0.715160, 0, 0,
            0, 0.119193, 0, 0 )

    def select_image(self):
        print("You selected the image {} !!".format(self.num))
        self.list_img.append(self.num)
        img = Image.open(self.electroptic_images[self.num]).resize((100, 100))
        self.photo = PIL.ImageTk.PhotoImage(img.convert("RGB", self.transform_matrix))
        self.b3.config(image=self.photo, width="100", height="100")

class Buttons3(tk.Frame):
    # this is a button class which shows the image on the buttons
    # therefore, we can select the images to plane fit
    def __init__(self, master, img_, num=0, root='', list_img=[]):
        super().__init__(master)
        self.master = master
        self.count = 0
        self.num = num
        self.list_img = list_img
        # list the images
        # self.electroptic_images = glob.glob(
        #     root + "/outside_out_pmvs/reconstruction_sequential/PMVS/visualize/*.jpg")
        # self.electroptic_images.sort(reverse=False)
        self.b3 = tk.Button(self.master, text="Pictures\n(click here)", command=self.select_image)
        # read the image and put it onto buttons
        self.photo_np = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        # self.photo = PIL.ImageTk.PhotoImage(Image.open(img_path).resize((100,100)))
        self.photo = Image.fromarray(self.photo_np)
        self.photo_imagetk = PIL.ImageTk.PhotoImage(self.photo)
        self.b3.config(image=self.photo_imagetk, width="100", height="100")
        # place the buttons
        self.b3.grid(row=int(self.num/5), column=self.num%5)
        # self.b3.pack()
        # self.pack()
        self.transform_matrix = (
            0, 0.157580, 0, 0,
            0, 0.715160, 0, 0,
            0, 0.119193, 0, 0 )

    def select_image(self):
        print("You selected the image {} !!".format(self.num))
        self.list_img.append(self.num)
        # img = Image.open(self.electroptic_images[self.num]).resize((100, 100))
        self.photo_imagetk = PIL.ImageTk.PhotoImage(self.photo.convert("RGB", self.transform_matrix))
        self.b3.config(image=self.photo_imagetk, width="100", height="100")

class ScrolledFrame(tk.Toplevel):

    def __init__(self, parent, vertical=True, horizontal=False):
        super().__init__(parent)
        self.geometry("600x500")


        # canvas for inner frame
        self._canvas = tk.Canvas(self)
        self._canvas.grid(row=0, column=0, sticky='news') # changed

        # create right scrollbar and connect to canvas Y
        self._vertical_bar = tk.Scrollbar(self, orient='vertical', command=self._canvas.yview)
        if vertical:
            self._vertical_bar.grid(row=0, column=1, sticky='ns')
        self._canvas.configure(yscrollcommand=self._vertical_bar.set)

        # create bottom scrollbar and connect to canvas X
        self._horizontal_bar = tk.Scrollbar(self, orient='horizontal', command=self._canvas.xview)
        if horizontal:
            self._horizontal_bar.grid(row=2, column=0, sticky='we')
        self._canvas.configure(xscrollcommand=self._horizontal_bar.set)

        # inner frame for widgets
        self.inner = tk.Frame(self._canvas, bg='red')
        self._window = self._canvas.create_window((0, 0), window=self.inner, anchor='nw')

        # autoresize inner frame
        self.columnconfigure(0, weight=1) # changed
        self.rowconfigure(0, weight=1) # changed

        # resize when configure changed
        self.inner.bind('<Configure>', self.resize)
        self._canvas.bind('<Configure>', self.frame_width)

    def frame_width(self, event):
        # resize inner frame to canvas size
        canvas_width = event.width
        self._canvas.itemconfig(self._window, width = canvas_width)

    def resize(self, event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))






# def fun():
#     img = cv2.imread("/home/ogam/Documents/mimari-proje/class1/flir_2_pipeline/images/FLIR0366.jpg")
#     cv2.namedWindow("img shower", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("img shower", 640, 480)
#     cv2.imshow("img shower", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# root = tk.Tk()
# app = Application(master=root)
# app.mainloop()

