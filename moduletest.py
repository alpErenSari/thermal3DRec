from plyProcess.basicGUI import Application
import os
import tkinter as tk

def get_parent_dir(directory):
    import os
    return os.path.dirname(directory)

# input_eval_dir_1 = "/home/ogam/Documents/mimari-proje/class1/flir_2_pipeline"
# input_eval_dir_1 = "/home/eren/Documents/mimari-proje/class1/flir_01_01_2019"
# input_eval_dir_1 = "/home/eren/Documents/mimari-proje/studio/flir-01-01-2019"
    #os.path.dirname(os.path.abspath(__file__))
# Checkout an OpenMVG image dataset with Git

# output_eval_dir = os.path.join(input_eval_dir_1, "outside_out_pmvs")
# input_eval_dir = os.path.join(input_eval_dir_1, "images")
# if not os.path.exists(output_eval_dir):
#   os.mkdir(output_eval_dir)
#
# input_dir = input_eval_dir
# output_dir = output_eval_dir
# print ("Using input dir  : ", input_dir)
# print ("      output_dir : ", output_dir)
#
# matches_dir = os.path.join(output_dir, "matches")
# camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")
# flir_e60_intrinsics = "1.9340e+03;0;1.0187e+03;0;1.8820e+03;710.7583;0;0;1"

# mvgRec = openMVGReconstruct(input_eval_dir_1)
# mvgRec.reconstruct()

# Create the ouput/matches folder if not present
# if not os.path.exists(matches_dir):
#   os.mkdir(matches_dir)

# print ("1. Intrinsics analysis")
# pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
#                                              "openMVG_main_SfMInit_ImageListing"),  "-i", input_dir, "-o", matches_dir, "-k", flir_e60_intrinsics, "-c", "3"] )
# pIntrisics.wait()
#
# print ("2. Compute features")
# pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-m", "SIFT", "-p", "ULTRA", "-f" , "1"] )
# pFeatures.wait()
#
# print ("2. Compute matches")
# pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-f", "1", "-n", "ANNL2"] )
# pMatches.wait()
#
# reconstruction_dir = os.path.join(output_dir,"reconstruction_sequential")
# print ("3. Do Incremental/Sequential reconstruction") #set manually the initial pair to avoid the prompt question
# pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_IncrementalSfM"),  "-i", matches_dir+"/sfm_data.json", "-m", matches_dir, "-o", reconstruction_dir] )
# pRecons.wait()

# print ("4. Control Point Registration")
# pControl = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "ui_openMVG_control_points_registration")] )
# pControl.wait()
#
# print ("5. Colorize Structure")
# pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/sfm_data.bin", "-o", os.path.join(reconstruction_dir,"colorized.ply")] )
# pRecons.wait()
#
# print ("6. Structure from Known Poses (robust triangulation)")
# pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),  "-i", reconstruction_dir+"/sfm_data.bin", "-m", matches_dir, "-o", os.path.join(reconstruction_dir,"robust.ply")] )
# pRecons.wait()
#
#
#
# print("7. Exporting to PMVS2")
# pPMVSout = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_openMVG2PMVS"),  "-i", reconstruction_dir+"/sfm_data.bin", "-o", reconstruction_dir] )
# pPMVSout.wait()
#
# print("\n8. Densing the cloud with PMVS2")
# pPMVS = subprocess.Popen( [os.path.join("/usr/local/bin/pmvs2"), reconstruction_dir+"/PMVS/", "pmvs_options.txt"] )
# pPMVS.wait()

# root = "/home/ogam/Documents/mimari-proje/class1/flir"

# thermal_root = os.path.join(input_eval_dir_1, "visualize_th")
# param = {
#     'root': input_eval_dir_1,
#     'thermal_root': thermal_root,
#     'rgb_intrinsic': os.path.join(input_eval_dir_1, "parameters/electroptic_camera.txt"),
#     'thermal_intrinsic': os.path.join(input_eval_dir_1, "parameters/thermal_camera.txt"),
#     'rot_diff_mat': os.path.join(input_eval_dir_1, "parameters/rotation_difference.txt"),
#     'trans_diff_': os.path.join(input_eval_dir_1, "parameters/translation_difference_2.txt"),
#     'begin_it': 43,
#     'iterationNo': 46,
#     'ransac_threshold': 0.05,
#     'gui_image_number': 36
# }

# th = thermal(param)
# ply_ = th.add_thermal_points_all()
# plane_list = [0,4,9,15,17]
# th.thermal_image_point_select(plane_list=plane_list)
# normals, errors = th.thermal_plane_fitting(plane_list=plane_list)
# normalized_normals = [-x/x[2] for x in normals]
# print("normal is ", normalized_normals)
# # print("normalized normal is ", -normal/normal[2])
# print("the error is ", errors)

os.chdir('/media/eren/1.0 TB/eren/mimari-proje/class1/class44_14_02019/Class44_7_70_2_COLMAP/')

root = tk.Tk()
root.title("SISER")
app = Application(master=root)
app.mainloop()

#ply = th.add_thermal_points_all()
#pr_mat = th.get_projection_matices()
#th.save_processed_point_cloud("magical_thermal_points")
#
# print(ply.shape)
# print(pr_mat[:2])
# print(ply)

