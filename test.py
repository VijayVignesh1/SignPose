import numpy as np
import os
import cv2
import torch
import json
import time
import glob
from models import BodyNetwork, HandNetwork
from utils import initialize_avatar, two_d_pose_image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

device="cuda"
body_model=BodyNetwork().to(device)
hand_model=HandNetwork().to(device)

checkpoint_body=ROOT_DIR + "/checkpoints/checkpoint.body.pth.tar"
checkpoint_hands=ROOT_DIR + "/checkpoints/checkpoint.hands.pth.tar"

if checkpoint_body:
    body_model.load_state_dict(torch.load(checkpoint_body)['state_dict'])
if checkpoint_hands:
    hand_model.load_state_dict(torch.load(checkpoint_hands)['state_dict'])

body_model.eval()
hand_model.eval()

folder=ROOT_DIR + "/test_inputs/"

armature, scene = initialize_avatar()

# loop through the directory and run the model
for image_file, json_file in zip(glob.glob(folder+"*.jpg"),glob.glob(folder+"*.json")):
    
    image_file=image_file.split("\\")[-1]
    json_file=json_file.split("\\")[-1]
    with open(folder+json_file) as f:
        data=json.load(f)
    img=cv2.imread(folder+image_file)

    # extract the openpose values
    pose=data["people"][0]["pose_keypoints_2d"]+data["people"][0]["hand_left_keypoints_2d"]+data["people"][0]["hand_right_keypoints_2d"]
    assert len(pose)==201
    pose=[m for l, m in enumerate(pose) if (l+1)%3]
    assert len(pose)==134
    pose=np.array(pose).reshape(67,2)

    
    normalize=np.array([img.shape[1], img.shape[0]])
    hand_input=np.vstack([pose[1:8],pose[25:]])
    input_poses=np.vstack([pose[:8],pose[15:19],pose[25:]])


    input_poses=input_poses/normalize
    input_poses=input_poses.flatten()
    input_poses=torch.FloatTensor(input_poses)

    hand_input=hand_input/normalize
    hand_input=hand_input.flatten()
    hand_input=torch.FloatTensor(hand_input)

    body_joints=['hips', 'spine', 'chest', 'upper-chest','shoulder.L', 'upperarm.L', 'forearm.L', 'hand.L',
                            'neck', 'head','shoulder.R', 'upperarm.R', 'forearm.R', 'hand.R']

    hand_joints=[ 'palm.04.L', 'little.01.L', 'little.02.L', 'little03.L', 'palm.03.L', 'ring.01.L', 'ring.02.L', 'ring.03.L', 
    'palm.02.L', 'middle.01.L', 'middle.02.L', 'middle.03.L', 'palm.01.L', 'index.01.L', 'index.02.L', 'index.03.L', 'thumb.01.L', 
    'thumb.02.L', 'thumb.03.L', 'palm.04.R', 'little.01.R', 'little.02.R', 
    'little03.R', 'palm.03.R', 'ring.01.R', 'ring.02.R', 'ring.03.R', 'palm.02.R', 'middle.01.R', 'middle.02.R', 'middle.03.R', 
    'palm.01.R', 'index.01.R', 'index.02.R', 'index.03.R', 'thumb.01.R', 'thumb.02.R', 'thumb.03.R']

    input_poses=input_poses.to(device)
    hand_input=hand_input.to(device)

    start=time.time()
    body_output=body_model(input_poses)
    hand_output=hand_model(hand_input)
    print("Time Taken: ", time.time()-start)

    body_output=dict(zip(body_joints,body_output.view(14,4).detach().tolist()))
    hand_output=dict(zip(hand_joints,hand_output.view(38,4).detach().tolist()))

    for i in hand_output:
        body_output[i]=hand_output[i]

    final_dict = dict(list(body_output.items()) + list(hand_output.items()))

    two_d_pose_image(final_dict, image_file, armature, scene)
