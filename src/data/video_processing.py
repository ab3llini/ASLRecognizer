import cv2
import os
import numpy as np


def unpack_video_train(vid_path, save_path, label, convention_number):
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    center = np.array(np.floor(np.array(np.shape(image)) / 2), dtype=np.int32)
    x1 = center[0] - 100
    x2 = center[0] + 100
    y1 = center[1] - 100
    y2 = center[1] + 100
    count = 0
    final_folder_path = os.path.join(save_path, label + "_" + str(convention_number))
    os.makedirs(final_folder_path)
    while success:
        cv2.imwrite(os.path.join(final_folder_path, "frame%d.jpg" % count), image[x1:x2, y1:y2, :])
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


if __name__ == '__main__':
    video = './../../dataset/videos/a.mp4'
    dest = './../../dataset/asl_alphabet_train'
    label = 'A'
    convention_number = 0
    unpack_video_train(video, dest, label, convention_number)

