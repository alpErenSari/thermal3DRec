import os
import subprocess
import sys
import time
from shutil import copyfile
from distutils.dir_util import copy_tree
from read_model import extract_poses_2_folder

class colmapReconstruct:
    def __init__(self, input_folder, image_folder):
        self.input_folder = input_folder
        self.image_folder = image_folder
        self.output_eval_dir = os.path.join(self.input_folder, "colmap_output")
        self.output_openmvg_dir = os.path.join(self.input_folder, "outside_out_pmvs","reconstruction_sequential","PMVS")
        self.ply_input_dir = os.path.join(self.output_eval_dir, "dense", "0", "fused.ply")
        self.output_openmvg_models = os.path.join(self.output_openmvg_dir, "models")
        self.ply_output_name = os.path.join(self.output_openmvg_models, "pmvs_options.txt.ply")
        self.colmap_image_path = os.path.join(self.output_eval_dir, "dense", "0", "images")
        self.output_openmvg_images_dir = os.path.join(self.output_openmvg_dir, "visualize")
        self.output_openmvg_txt_dir = os.path.join(self.output_openmvg_dir, "txt")

        if not os.path.exists(self.output_eval_dir):
            os.mkdir(self.output_eval_dir)

        if not os.path.exists(self.output_openmvg_dir):
            os.makedirs(self.output_openmvg_dir)

        if not os.path.exists(self.output_openmvg_models):
            os.mkdir(self.output_openmvg_models)

        if os.path.exists(self.ply_input_dir):
            copyfile(self.ply_input_dir, self.ply_output_name)

        if not os.path.exists(self.output_openmvg_images_dir):
            copy_tree(self.colmap_image_path, self.output_openmvg_images_dir)

        extract_poses_2_folder(self.output_eval_dir, self.output_openmvg_txt_dir, file_ext='.bin')




    def reconstruct(self):
        print("1. Automatic Reconstruction via COLMAP")
        pAutomatic = subprocess.Popen(
            ["colmap", "automatic_reconstructor", "--workspace_path", self.output_eval_dir,
             "--image_path", self.image_folder])
        pAutomatic.wait()

        if os.path.exists(self.ply_input_dir):
            copyfile(self.ply_input_dir, self.ply_output_name)
        else:
            print("Dense point cloud cannot be found")
