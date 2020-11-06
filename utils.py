import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import time

image_size = (170,170)
img_path = 'imageNet_val/ILSVRC2010_images_val/val'
test_path = 'test_imgs/'


num_train_samples = 42500
num_val_samples = 7500

def resize_images(img_n):
    img_ = os.path.join(img_path,img_n)
    img = cv2.imread(img_)
    img = cv2.resize(img, image_size)
    cv2.imwrite(os.path.join(img_path , img_n),img)

def load_dat():
    pbar = tqdm(total=50000)


    f = open('imageNet_val/ILSVRC2010_validation_ground_truth.txt','r')
    dat_y = []
    # tst = 100
    # i = 0
    for lable in f:
        # i += 1
        # if i> tst:
        #     break
        dat_y.append(lable)

    dat_x = []
    # i = 0
    for f_n in os.listdir(img_path):
        # i += 1
        # if i > tst:
        #     break
        img_ = os.path.join(img_path, f_n)
        img = cv2.imread(img_)
        dat_x.append(img)
        pbar.update(1)

    dat_x = np.array(dat_x)

    dat_y = np.array(dat_y)
    dat_y = dat_y.astype('int32')
    # X_train, X_test, y_train, y_test = train_test_split(dat,test_size=0.15)
    return train_test_split(dat_x,dat_y,test_size=0.15)


def main():
    pbar = tqdm(total=50000)
    for f_n in os.listdir(img_path):
        resize_images(f_n)
        pbar.update(1)

if __name__ == '__main__':
    main()


# X_train, X_test, y_train, y_test = load_dat()
