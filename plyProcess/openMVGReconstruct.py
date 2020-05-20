# Indicate the openMVG binary directory
OPENMVG_SFM_BIN = "/home/eren/Downloads/openMVG/openMVG_Build/Linux-x86_64-Release"

# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/eren/Downloads/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"

import os
import subprocess
import sys
import time

class openMVGReconstruct:
    def __init__(self, input_folder, image_folder, isThermal=2):
        self.input_eval_dir_1 = input_folder
        self.output_eval_dir = os.path.join(self.input_eval_dir_1, "outside_out_pmvs")
        # self.input_eval_dir = os.path.join(self.input_eval_dir_1, "images")
        self.input_eval_dir = image_folder
        self.isThermal = isThermal
        
    def get_parent_dir(self, directory):
        import os
        return os.path.dirname(directory)

    def reconstruct(self):

        start = time.time()

        if not os.path.exists(self.output_eval_dir):
          os.mkdir(self.output_eval_dir)
        
        self.input_dir = self.input_eval_dir
        self.output_dir = self.output_eval_dir
        print ("Using input dir  : ", self.input_dir)
        print ("      self.output_dir : ", self.output_dir)
        
        self.matches_dir = os.path.join(self.output_dir, "matches")
        self.camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")
        self.flir_e60_intrinsics = "1.9340e+03;0;1.0187e+03;0;1.8820e+03;710.7583;0;0;1"
        self.xiaomi_koral_intrinsics = "2842.36192442258;0;1995.16449777960;0;2840.09170809351;1559.94151455342;0;0;1"
        
        # Create the ouput/matches folder if not present
        if not os.path.exists(self.matches_dir):
          os.mkdir(self.matches_dir)
        
        print ("1. Intrinsics analysis")
        if(self.isThermal is 1):
            print("Nikon D90 selected")
            pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
                                                         "openMVG_main_SfMInit_ImageListing"),
                                           "-i", self.input_dir, "-o", self.matches_dir, "-c", "3", "-d", self.camera_file_params] )
            pIntrisics.wait()
        elif(self.isThermal is 2):
            print("Koral Xiaomi selected")
            pIntrisics = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN,
                                                        "openMVG_main_SfMInit_ImageListing"), "-i", self.input_dir,
                                           "-o", self.matches_dir, "-k", self.xiaomi_koral_intrinsics, "-c", "3"])

            pIntrisics.wait()
        else:
            print("Thermal Camera selected")
            pIntrisics = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN,
                                                        "openMVG_main_SfMInit_ImageListing"), "-i", self.input_dir, "-o",
                                           self.matches_dir, "-k", self.flir_e60_intrinsics, "-c", "3"])
            pIntrisics.wait()

        print ("2. Compute features")
        if(self.isThermal is 2):
            precision = "ULTRA"
        else:
            precision = "NORMAL"
        pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", self.matches_dir+"/sfm_data.json", "-o", self.matches_dir, "-m", "SIFT", "-p", precision, "-f" , "1"] )
        pFeatures.wait()

        print ("2. Compute matches")
        pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", self.matches_dir+"/sfm_data.json", "-o", self.matches_dir, "-f", "1", "-n", "ANNL2"] )
        pMatches.wait()
        
        self.reconstruction_dir = os.path.join(self.output_dir,"reconstruction_sequential")
        print ("3. Do Incremental/Sequential reconstruction") #set manually the initial pair to avoid the prompt question
        pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_IncrementalSfM"),  "-i", self.matches_dir+"/sfm_data.json", "-m", self.matches_dir, "-o", self.reconstruction_dir] )
        pRecons.wait()

        end = time.time()
        dur1 = end - start
        
        print ("4. Control Point Registration")
        pControl = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "ui_openMVG_control_points_registration")] )
        pControl.wait()

        start = time.time()
        
        print ("5. Colorize Structure")
        pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", self.reconstruction_dir+"/sfm_data.bin", "-o", os.path.join(self.reconstruction_dir,"colorized.ply")] )
        pRecons.wait()
        
        print ("6. Structure from Known Poses (robust triangulation)")
        pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),  "-i", self.reconstruction_dir+"/sfm_data.bin", "-m", self.matches_dir, "-o", os.path.join(self.reconstruction_dir,"robust.ply")] )
        pRecons.wait()
        
        print("7. Exporting to PMVS2")
        pPMVSout = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_openMVG2PMVS"),  "-i", self.reconstruction_dir+"/sfm_data.bin", "-o", self.reconstruction_dir] )
        pPMVSout.wait()

        if (self.isThermal is not 3):
            with open(os.path.join(self.reconstruction_dir, 'PMVS', 'pmvs_options.txt'), 'r') as f:
                file_txt = f.readlines()
            file_txt[1] = 'csize 4 \n'
            with open(os.path.join(self.reconstruction_dir, 'PMVS', 'pmvs_options.txt'), 'w') as f:
                f.writelines(file_txt)

        
        print("\n8. Densing the cloud with PMVS2")
        pPMVS = subprocess.Popen( ["/usr/local/bin/pmvs2", self.reconstruction_dir+"/PMVS/", "pmvs_options.txt"] )
        pPMVS.wait()

        end = time.time()

        dur = dur1 + end - start

        print("TOTAL 3D Reconstruction process took {} seconds".format(dur))
