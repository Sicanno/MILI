import os
import sys

from tqdm import tqdm

import argparse

import cv2 as cv
import numpy
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-loc', '--mainFolder', type=str, default='/mnt/disk1/2020/multi/data/mpi_inf_3dhp/S1/Seq1_img')
args = parser.parse_args()

img_main_folder_path = args.mainFolder
para_main_folder_path = img_main_folder_path

list_img_folder_name = os.listdir(img_main_folder_path)

for folder_count in range(len(list_img_folder_name)) :
    img_folder_name = list_img_folder_name[folder_count]
    img_folder_path = os.path.join(img_main_folder_path, img_folder_name)
    
    para_folder_name = img_folder_name + '_para'
    para_folder_path = os.path.join(img_main_folder_path, para_folder_name)
    
    string_shell = 'python3 ./tools/demo.py' +  \
                    ' --config=' + str('configs/smpl/tune.py') + \
                    ' --image_folder=' + str(img_folder_path) + \
                    ' --output_folder=' + str(para_folder_path) + \
                    ' --ckpt ' + str('data/checkpoint.pt')
    
    
    print(string_shell)
    os.system(string_shell)
    