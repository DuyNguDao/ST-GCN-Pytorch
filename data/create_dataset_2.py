"""
This script to extract skeleton joints position and score.
- This 'annot_folder' is a action class and bounding box for each frames that came with dataset.
    Should be in format of [frame_idx, action_cls, xmin, ymin, xmax, ymax]
        Use for crop a person to use in pose estimation model.
- If have no annotation file you can leave annot_folder = '' for use Detector model to get the
    bounding box.
"""

import os
import cv2
import time
import torch
import pandas as pd
import numpy as np


import sys
sys.path.append('/media/ngocthien/DATA/DO_AN_TOT_NGHIEP/TRAIN')

from yolov7.pose import yolov7_pose
from yolov7.utils.plots import plot_skeleton_kpts

model = yolov7_pose('./yolov7/weight/yolov7-w6-pose.pt')

save_path = './outputs_data/ouput_create_2/pose_and_score.csv'
annot_file = './datasets/dataset_create_2/annotation.csv'
video_folder = './datasets/dataset_create_2/videos'

columns = ['video', 'frame', 'nose_x','nose_y','nose_s','left_eye_x','_left_eye_y','left_eye_s','right_eye_x','right_eye_y','right_eye_s','left_ear_x','left_ear_y','left_ear_s','right_ear_x','right_ear_y','right_ear_s','left_shoulder_x','left_shoulder_y','left_shoulder_s','right_shoulder_x','right_shoulder_y','right_shoulder_s','left_elbow_x','left_elbow_y','left_elbow_s','right_elbow_x','right_elbow_y','right_elbow_s','left_wrist_x','left_wrist_y','left_wrist_s','right_wrist_x','right_wrist_y','right_wrist_s','left_hip_x','left_hip_y','left_hip_s','right_hip_x','right_hip_y','right_hip_s','left_knee_x','left_knee_y','left_knee_s','right_knee_x','right_knee_y','right_knee_s','left_ankle_x','left_ankle_y','left_ankle_s','right_ankle_x','right_ankle_y','right_ankle_s','label']
class_names = ['Standing','Stand up', 'Sitting','Sit down','Lying Down','Walking','Fall Down']
frame_size = [640,640]

def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy

def YL2XY(kpts,steps=3,thresh=0.01):

    result = np.zeros([17,3])
    cf = True
    num_kpts = len(kpts) // steps
    for kid in range(num_kpts):
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if (conf < thresh) and (kid in [5,6,11,12,13,14]):
                    cf = False
                result[kid,0] = x_coord
                result[kid,1] = y_coord
                result[kid,2] = conf
    return result,cf


annot = pd.read_csv(annot_file)
vid_list = annot['video'].unique()
for vid in vid_list:
    print(f'Process on: {vid}')
    df = pd.DataFrame(columns=columns)
    cur_row = 0

    # Pose Labels.
    frames_label = annot[annot['video'] == vid].reset_index(drop=True)

    cap = cv2.VideoCapture(os.path.join(video_folder, vid))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fps_time = 0
    i = 1
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,[640,640])
            img = frame.copy()
            cls_idx = int(frames_label[frames_label['frame'] == i]['label'])
            

            image,keypoints = model.predict1(frame)
            if keypoints.shape[0] > 0:
                bb = np.array(keypoints[0,1:5])
                result,cf = YL2XY(kpts=keypoints[0,6:],steps=3,thresh=0.01)
                if cf:
                    pt_norm = normalize_points_with_size(result,
                                                        frame_size[0], frame_size[1])
                    row = [vid, i, *pt_norm.flatten().tolist(), cls_idx]
                    scr = result[:,2].mean()

                    # VISUALIZE.
                    plot_skeleton_kpts(img, keypoints[0, 6:].T, 3)
                    bb = bb.astype(int)
                    xmin, ymin = (bb[0]-int(bb[2]/2)), (bb[1]-int(bb[3]/2))
                    xmax, ymax = (bb[0]+int(bb[2]/2)), (bb[1]+int(bb[3]/2))
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (125, 255,255 ), 2)
                    
                    img = cv2.putText(img, 'Pose: {}, Score: {:.4f}'.format( class_names[cls_idx], scr),
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0), 2)

                else:
                    row = [vid, i, *[np.nan] * (17 * 3), cls_idx]
                    scr = 0.0
            else:
                row = [vid, i, *[np.nan] * (17 * 3), cls_idx]
                scr = 0.0

            df.loc[cur_row] = row
            cur_row += 1
            i += 1
            cv2.putText(img,vid+'   Frame:' +str(i),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0 ), 2)
            img = img[:, :, ::-1]
            fps_time = time.time()
            

            cv2.imshow('frame', img)
            key =cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('p'):
                key =cv2.waitKey(0)
                if key & 0xFF == ord('q'):
                    break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, mode='w', index=False)