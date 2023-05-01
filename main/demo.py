import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
from config import cfg
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--result_path', type=str, default='output/test')
    parser.add_argument('--ckpt_idx', type=int, default=0)
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', type=str, default='na')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--output_folder', type=str, default='output')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_path = osp.join('../output',args.result_path, 'code', 'config_base.py')
    ckpt_path = osp.join('../output', args.result_path, 'model_dump', f'snapshot_{int(args.ckpt_idx)}.pth.tar')
    # config_path = '/mnt/cache/yinwanqi/01-project/osx/main/config/config_base.py'
    # ckpt_path = '/mnt/cache/yinwanqi/01-project/osx/pretrained_models/osx_l_agora.pth.tar'

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(args.testset, args.agora_benchmark, ckpt_path)
    cfg.update_config(args.num_gpus, args.exp_name)
    cudnn.benchmark = True

    # load model
    from common.base import Demoer
    demoer = Demoer()
    demoer._make_model()
    from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
    from common.utils.vis import render_mesh, save_obj
    from common.utils.human_models import smpl_x
    # model_path = args.pretrained_model_path
    # assert osp.exists(model_path), 'Cannot find model at ' + model_path
    # print('Load checkpoint from {}'.format(model_path))

    demoer.model.eval()

    start = int(args.start)
    end = int(args.end)
    for frame in tqdm(range(start, end)):
        img_path = f'{args.img_path}_{int(frame):05d}.jpg'
        # prepare input image
        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]
        os.makedirs(args.output_folder, exist_ok=True)

        #### detect human bbox with yolov5s 
        # detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # with torch.no_grad():
        #     results = detector(original_img)
        # person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
        # class_ids, confidences, boxes = [], [], []
        # for detection in person_results:
        #     x1, y1, x2, y2, confidence, class_id = detection.tolist()
        #     class_ids.append(class_id)
        #     confidences.append(confidence)
        #     boxes.append([x1, y1, x2 - x1, y2 - y1])
        # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # vis_img = original_img.copy()
        # for num, indice in enumerate(indices):
            # bbox = boxes[indice]  # x,y,h,w
        
        ### HARDCODE for testing
        vis_img = original_img.copy()
        for num, indice in enumerate(range(1)):
            bbox = [40, 10, 1000, 1900] # xmin, ymin, width, height


            bbox = process_bbox(bbox, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
            mesh = mesh[0]
            # import pdb;pdb.set_trace()

            # save mesh
            save_obj(mesh, smpl_x.face, os.path.join(args.output_folder, f'person_{frame:05}.obj'))

            # render mesh
            focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
            princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=True)

        # save rendered image
        frame_name = img_path.split('/')[-1]
        cv2.imwrite(os.path.join(args.output_folder, f'{frame_name}'), vis_img[:, :, ::-1])

if __name__ == "__main__":
    main()