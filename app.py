import os
import sys
import os.path as osp
from pathlib import Path
import cv2
import gradio as gr
import torch
import math
import spaces
from huggingface_hub import hf_hub_download
try:
    import mmpose
except:
    os.system('pip install /home/user/app/main/transformer_utils')
hf_hub_download(repo_id="caizhongang/SMPLer-X", filename="smpler_x_h32.pth.tar", local_dir="/home/user/app/pretrained_models")
os.system('cp -rf /home/user/app/assets/conversions.py /usr/local/lib/python3.10/site-packages/torchgeometry/core/conversions.py')
DEFAULT_MODEL='smpler_x_h32'
OUT_FOLDER = '/home/user/app/demo_out'
os.makedirs(OUT_FOLDER, exist_ok=True)
num_gpus = 1 if torch.cuda.is_available() else -1
print("!!!", torch.cuda.is_available())      
print(torch.cuda.device_count())   
print(torch.version.cuda)  
index = torch.cuda.current_device()
print(index)  
print(torch.cuda.get_device_name(index))
# from main.inference import Inferer
# inferer = Inferer(DEFAULT_MODEL, num_gpus, OUT_FOLDER)

@spaces.GPU(enable_queue=True, duration=300)
def infer(video_input, in_threshold=0.5, num_people="Single person", render_mesh=False):
    from main.inference import Inferer
    inferer = Inferer(DEFAULT_MODEL, num_gpus, OUT_FOLDER)
    os.system(f'rm -rf {OUT_FOLDER}/*')
    multi_person = False if (num_people == "Single person") else True
    cap = cv2.VideoCapture(video_input)
    fps = math.ceil(cap.get(5))
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = osp.join(OUT_FOLDER, f'out.m4v')
    final_video_path = osp.join(OUT_FOLDER, f'out.mp4')
    video_output = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    success = 1
    frame = 0
    while success:
        success, original_img = cap.read()
        if not success:
            break
        frame += 1
        img, mesh_paths, smplx_paths = inferer.infer(original_img, in_threshold, frame, multi_person, not(render_mesh))
        video_output.write(img)
        yield img, None, None, None
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
    os.system(f'ffmpeg -i {video_path} -c copy {final_video_path}')

    #Compress mesh and smplx files
    save_path_mesh = os.path.join(OUT_FOLDER, 'mesh')
    save_mesh_file = os.path.join(OUT_FOLDER, 'mesh.zip')
    os.makedirs(save_path_mesh, exist_ok= True)
    save_path_smplx = os.path.join(OUT_FOLDER, 'smplx')
    save_smplx_file = os.path.join(OUT_FOLDER, 'smplx.zip')
    os.makedirs(save_path_smplx, exist_ok= True)
    os.system(f'zip -r {save_mesh_file} {save_path_mesh}')
    os.system(f'zip -r {save_smplx_file} {save_path_smplx}')
    yield img, video_path, save_mesh_file, save_smplx_file

TITLE = '''<h1 align="center">SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation</h1>'''
VIDEO = '''
<center><iframe width="960" height="540" 
src="https://www.youtube.com/embed/DepTqbPpVzY?si=qSeQuX-bgm_rON7E"title="SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen>
</iframe>
</center><br>'''
DESCRIPTION = '''
<b>Official Gradio demo</b> for <a href="https://caizhongang.com/projects/SMPLer-X/"><b>SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation</b></a>.<br>
<p>
Note: You can drop a video at the panel (or select one of the examples) 
to obtain the 3D parametric reconstructions of the detected humans.
</p>
'''

with gr.Blocks(title="SMPLer-X", css=".gradio-container") as demo:

    gr.Markdown(TITLE)
    gr.HTML(VIDEO)
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input video", elem_classes="video")
            threshold = gr.Slider(0, 1.0, value=0.5, label='BBox detection threshold')
        with gr.Column(scale=2):
            num_people = gr.Radio(
                choices=["Single person", "Multiple people"],
                value="Single person",
                label="Number of people",
                info="Choose how many people are there in the video. Choose 'single person' for faster inference.",
                interactive=True,
                scale=1,)
            gr.HTML("""<br/>""")
            mesh_as_vertices = gr.Checkbox(
                label="Render as mesh", 
                info="By default, the estimated SMPL-X parameters are rendered as vertices for faster visualization. Check this option if you want to visualize meshes instead.",
                interactive=True,
                scale=1,)

            send_button = gr.Button("Infer")
    gr.HTML("""<br/>""")

    with gr.Row():
        with gr.Column():
            processed_frames = gr.Image(label="Last processed frame")
            video_output = gr.Video(elem_classes="video")
        with gr.Column():
            meshes_output = gr.File(label="3D meshes")
            smplx_output = gr.File(label= "SMPL-X models")
    # example_images = gr.Examples([])
    send_button.click(fn=infer, inputs=[video_input, threshold, num_people, mesh_as_vertices], outputs=[processed_frames, video_output, meshes_output, smplx_output])
    # with gr.Row():
    example_videos = gr.Examples([
        ['/home/user/app/assets/01.mp4'], 
        ['/home/user/app/assets/02.mp4'], 
        ['/home/user/app/assets/03.mp4'],
        ['/home/user/app/assets/04.mp4'],
        ['/home/user/app/assets/05.mp4'], 
        ['/home/user/app/assets/06.mp4'], 
        ['/home/user/app/assets/07.mp4'],
        ['/home/user/app/assets/08.mp4'],
        ['/home/user/app/assets/09.mp4'],
        ], 
        inputs=[video_input, 0.5])

#demo.queue()
demo.queue().launch(debug=True)
