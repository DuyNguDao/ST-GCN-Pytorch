import os
import cv2
import numpy as np
import pandas as pd
import sys

class_names = ['Standing','Stand up', 'Sitting','Sit down','Lying Down','Walking','Fall Down','No action']

video_folder = '/media/ngocthien/DATA/DO_AN_TOT_NGHIEP/DATA/FallDataset/Coffee_room_01/Videos'
annot_file_2 = '/media/ngocthien/DATA/DO_AN_TOT_NGHIEP/DATA/FallDataset/Coffee_room_01/coffee_room_01_label_2.csv'


video_list = sorted(os.listdir(video_folder))
cols = ['video', 'frame', 'label']
df = pd.DataFrame(columns=cols)


for index_video_to_play in range(len(video_list)):

    video_file = os.path.join(video_folder, video_list[index_video_to_play])
    print(os.path.basename(video_file))

    cap = cv2.VideoCapture(video_file)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video = np.array([video_list[index_video_to_play]] * frames_count)
    frame_idx = np.arange(1, frames_count + 1)
    label = np.array([0] * frames_count)

    k = 7 #No action
    i = 0
    while True:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if ret:

            frame = cv2.resize(frame,(1000, 640))

            #show button
            cv2.putText(frame, 'Video: {}     Total_frames: {}        Frame: {}       '.format(video_list[index_video_to_play],frames_count,i),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            for c in range(len(class_names)):
                cv2.putText(frame,'{}: {}'.format(class_names[c],c),(10, 300 +30*c), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame,'Back:   a', (10, 530),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imshow('frame', frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                sys.exit("Exit!")
            elif key == ord('a') and i >0:#Trở lại
                i -= 1
                print("Back at frame {}: {}".format(i,class_names[label[i].item()]))
            elif (key - 48) in [0,1,2,3,4,5,6,7]:
                label[i] = key - 48
                cls_name = class_names[key - 48]
                print("frame {}: {}".format(i,cls_name))
                i += 1
        else:
            break
        
    rows = np.stack([video, frame_idx, label], axis=1)
    df = df.append(pd.DataFrame(rows, columns=cols),ignore_index=True)
df.to_csv(annot_file_2,index=False)
cap.release()
cv2.destroyAllWindows()