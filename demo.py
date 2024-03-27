import os
import sys
import os.path as osp
import argparse
from pathlib import Path
import cv2
import torch
import math
import mmpose
import shutil
import time
from OpenGL import GL
from OpenGL.GL import *
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
try:
    import mmpose
except:
    os.system('pip install main/transformer_utils')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_verts', action="store_true")
    parser.add_argument('--multi_person', action="store_true")
    parser.add_argument('--in_threshold', type=float, default=0.5)
    parser.add_argument('--output_folder', type=str, default='demo_out')
    parser.add_argument('--pretrained_model', type=str, default='smpler_x_h32')
    parser.add_argument('--input_video', type=str, default='')
    args = parser.parse_args()
    return args

def infer():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    num_gpus = 1 if torch.cuda.is_available() else -1

    from main.inference import Inferer
    inferer = Inferer(args.pretrained_model, num_gpus, args.output_folder)

    cap = cv2.VideoCapture(args.input_video)
    fps = math.ceil(cap.get(5))
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = osp.join(args.output_folder, f'out.m4v')
    final_video_path = osp.join(args.output_folder, f'out.mp4')
    video_output = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    success = 1
    frame = 0
    while success:
        success, original_img = cap.read()
        if not success:
            break
        frame += 1
        img, mesh_paths, smplx_paths = inferer.infer(original_img, args.in_threshold, frame, args.multi_person, args.show_verts)
        video_output.write(img)
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
    os.system(f'ffmpeg -i {video_path} -c copy {final_video_path}')

    #Compress mesh and smplx files
    save_path_mesh = os.path.join(args.output_folder, 'mesh')
    save_mesh_file = os.path.join(args.output_folder, 'mesh.zip')
    os.makedirs(save_path_mesh, exist_ok= True)
    save_path_smplx = os.path.join(args.output_folder, 'smplx')
    save_smplx_file = os.path.join(args.output_folder, 'smplx.zip')
    os.makedirs(save_path_smplx, exist_ok= True)
    os.system(f'zip -r {save_mesh_file} {save_path_mesh}')
    os.system(f'zip -r {save_smplx_file} {save_path_smplx}')

if __name__ == "__main__":
    main()

