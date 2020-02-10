import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import pyrealsense2 as rs
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
print(ROOT_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs

from model_pose_ren import ModelPoseREN
import util
from util import get_center_fast as get_center

img_array = '/home/boonyew/Documents/HandJointRehab/test_seq2.npy'
def read_images(file_dir,depth_scale):
    #if not depth_frame:
    #    return None
    # Convert images to numpy array
    depth_frames = np.load(file_dir)
    depth_frames_a = np.asarray([img*depth_scale*1000 for img in depth_frames],dtype=np.float32)
    # depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)
    # depth = depth_image * depth_scale * 1000
    return depth_frames_a

def show_results(img, results, cropped_image, dataset):
    img = np.minimum(img, 1500)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img*255)
    # draw cropped image
    img[:96, :96] = (cropped_image+1)*255/2
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (0, 0), (96, 96), (255, 0, 0), thickness=2)
    img_show = util.draw_pose(dataset, img, results)
    return img_show

def main():
    # intrinsic paramters of Intel Realsense SR300
    fx, fy, ux, uy = 628.668, 628.668, 311.662, 231.571
    depth_scale = 0.0010000000474974513
    # paramters
    dataset = 'hands17'
    if len(sys.argv) == 2:
        dataset = sys.argv[1]
    print(dataset)
    lower_ = 1
    upper_ = 435

    # init hand pose estimation model
    try:
        hand_model = ModelPoseREN(dataset,
            lambda img: get_center(img, lower=lower_, upper=upper_),
            param=(fx, fy, ux, uy), use_gpu=True)
    except:
        print('Model not found')
    # for msra dataset, use the weights for first split
    if dataset == 'msra':
        hand_model.reset_model(dataset, test_id = 0)
    # realtime hand pose estimation loop
    frames = read_images(img_array,depth_scale)
    # preprocessing depth
    # # training samples are left hands in icvl dataset,
    # # right hands in nyu dataset and msra dataset,
    # # for this demo you should use your right hand
    if dataset == 'icvl':
        frames = frames[:, ::-1]  # flip
    # get hand pose
    predicted = []
    f = open('results.txt','w')
    for idx,depth in enumerate(frames):
        depth = frames[idx,:,:]
#        depth = np.rot90(depth,2)
        depth[depth == 0] = depth.max()
        results, cropped_image = hand_model.detect_image(depth)
        img_show = show_results(depth, results, cropped_image, dataset)
        # cv2.imshow('result', img_show)
        cv2.imwrite('result_{}.png'.format(idx), img_show)
        f.write('image_{}.png'.format(idx))
        for r in results:
            for i in r:
	            f.write(' %s' % i)
        f.write('\n')
        #print(results)
        predicted.append(results)
    f.close()
if __name__ == '__main__':
    main()

