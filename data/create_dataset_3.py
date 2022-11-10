
"""
This script to create dataset and labels by clean off some NaN, do a normalization,
label smoothing and label weights by scores.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class_names = ['Standing','Stand up', 'Sitting','Sit down','Lying Down','Walking','Fall Down']
# main_parts = ['nose_x','nose_y','nose_s','left_eye_x','_left_eye_y','left_eye_s','right_eye_x','right_eye_y','right_eye_s','left_ear_x','left_ear_y','left_ear_s','right_ear_x','right_ear_y','right_ear_s','left_shoulder_x','left_shoulder_y','left_shoulder_s','right_shoulder_x','right_shoulder_y','right_shoulder_s','left_elbow_x','left_elbow_y','left_elbow_s','right_elbow_x','right_elbow_y','right_elbow_s','left_wrist_x','left_wrist_y','left_wrist_s','right_wrist_x','right_wrist_y','right_wrist_s','left_hip_x','left_hip_y','left_hip_s','right_hip_x','right_hip_y','right_hip_s','left_knee_x','left_knee_y','left_knee_s','right_knee_x','right_knee_y','right_knee_s','left_ankle_x','left_ankle_y','left_ankle_s','right_ankle_x','right_ankle_y','right_ankle_s']
# main_idx_parts = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-1]  # 1.5

main_parts = ['left_shoulder_x','left_shoulder_y','left_shoulder_s','right_shoulder_x','right_shoulder_y','right_shoulder_s','left_hip_x','left_hip_y','left_hip_s','right_hip_x','right_hip_y','right_hip_s','left_knee_x','left_knee_y','left_knee_s','right_knee_x','right_knee_y','right_knee_s']
main_idx_parts = [5,6,11,12,13,14] #1.5 score


csv_pose_file = '/media/ngocthien/DATA/DO_AN_TOT_NGHIEP/DATA/FallDataset/annot/dataset.csv'
save_path_train = '/media/ngocthien/DATA/DO_AN_TOT_NGHIEP/DATA/FallDataset/annot/train.pkl'
save_path_test = '/media/ngocthien/DATA/DO_AN_TOT_NGHIEP/DATA/FallDataset/annot/test.pkl'

# Params.
smooth_labels_step = 8
n_frames = 30
skip_frame = 1

annot = pd.read_csv(csv_pose_file)

# Remove NaN.
# nan elements are true
idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
idx = np.where(idx)[0] # Search for row contain at lease one value nan

annot = annot.drop(idx) # remove rows contain at lease one value nan


# One-Hot Labels.
label_onehot = pd.get_dummies(annot['label'])
annot = annot.drop('label', axis=1).join(label_onehot)

cols = label_onehot.columns.values# [0,1,2,3,4,5,6]

feature_set = np.empty((0, n_frames, 17, 3))
labels_set = np.empty((0, len(cols)))

def scale_pose(xy):
    """
    Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


def seq_label_smoothing(labels, max_step=10):
    # lam min khoang giao nhau giua cac hanh dong khac nhau
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels)
    min_val = np.min(labels)
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue
        
        #tim kiếm sự khác biệt giữa nhãn của frame hiện tại với nhãn của 8 frame tiếp theo
        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0] #vị trí có sự thay đổi nhãn so với nhãn hiện tại trong chuỗi 8 frame
            remain_step = steps   
            target_label = np.argmax(labels[i + remain_step]) #nhãn khác so với nhãn hiện tại
            active_label = np.argmax(labels[i])               #nhãn hiện tại
    return labels
def plot_total_label(labels_set,cols):
    # label with labels_set max
    label =np.argmax(labels_set,axis=1)

    #find index in cols for class_names: ['Standing', 'Walking', 'Sitting', 'Lying Down','Stand up', 'Sit down', 'Fall Down']
    label_idx=[]  # tìm
    cols=list(cols)
    for i in range (len(class_names )):
        if i not in cols:
            label_idx.append(None)
        else:
            label_idx.append(cols.index(i))

    #calculate total number of frames for each class
    values=np.zeros((len(class_names),),dtype=int)
    for i in range(len(values)):
        values[i]=sum(label==label_idx[i])

    #plot total number frames for each class
    names = ['Standing','Stand up', 'Sitting','Sit down','Lying Down','Walking','Fall Down']
    n= list(np.arange(len(names)))
    plt.figure(figsize=(20,25))
    plt.bar(names, values)
    plt.title('total number of labels')
    plt.xlabel('classes')
    plt.ylabel("total labels")
    for i in range(len(values)):
        plt.annotate(str(values[i]), xy=(n[i],values[i]), ha='center', va='bottom')
    plt.show()


vid_list = annot['video'].unique()
for vid in vid_list:
    print(f'Process on: {vid}')
    # reset_index: đặt lại index do những cột đã bị loại bỏ những giá trị nan
    data = annot[annot['video'] == vid].reset_index(drop=True).drop(columns='video')

    # Label Smoothing.
    #được sử dụng khi hàm mất mát là cross-entropy và softmax làm lớp đầu ra
    #Giúp hạn chế hiện tượng over fit
    esp = 0.1
    data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1) #label smoothing cho 1 frame
    data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step)

    # Separate continuous frames.
    frames = data['frame'].values
    frames_set = []
    fs = [0]
    #neu detect miss 10 frame thi tiep tuc mot set moi
    for i in range(1, len(frames)):
        if frames[i] < frames[i-1] + 10:
            fs.append(i)
        else:
            frames_set.append(fs)
            fs = [i]
    frames_set.append(fs)

    for fs in frames_set:
        xys = data.iloc[fs, 1:-len(cols)].values.reshape(-1, 17, 3)
        # Scale pose normalize.
        xys[:, :, :2] = scale_pose(xys[:, :, :2])
        # Add center point. trung diem cua hai vai
        # xys = np.concatenate((xys, np.expand_dims((xys[:, 5, :] + xys[:, 6, :]) / 2, 1)), axis=1)

        # Weighting main parts score.
        scr = xys[:, :, -1].copy()
        scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)
        # Mean score.
        scr = scr.mean(1)

        # Targets.
        lb = data.iloc[fs, -len(cols):].values
        # Apply points score mean to all labels.
        lb = lb * scr[:, None]

        # 30 frames
        for i in range(xys.shape[0] - n_frames):
            feature_set = np.append(feature_set, xys[i:i+n_frames][None, ...], axis=0)
            labels_set = np.append(labels_set, lb[i:i+n_frames].mean(0)[None, ...], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(feature_set, labels_set, test_size=0.2, random_state=0)
    with open(save_path_train, 'wb') as f:
        pickle.dump((X_train,y_train), f)
    with open(save_path_test, 'wb') as f:
        pickle.dump((X_test,y_test), f)
    '''
    with open(save_path, 'wb') as f:
        pickle.dump((feature_set, labels_set), f)
    '''



plot_total_label(y_train,cols)
plt.figure()
plot_total_label(y_test,cols)



