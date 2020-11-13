import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import time

image_size = (80,80)
# training images path
img_path = 'imageNet_val/ILSVRC2010_images_val/val'
# validation images path
img_val_path = '/home/jinyu/Downloads/ILSVRC2010_images_test/test'

img2resize_path = ['/home/jinyu/Downloads/ILSVRC2010_images_test/test','imageNet_val/ILSVRC2010_images_val/val']




def resize_images(img_n,i):
    img_ = os.path.join(img_path[i],img_n)
    img = cv2.imread(img_)
    img = cv2.resize(img, image_size)
    cv2.imwrite(os.path.join(img_path[i] , img_n),img)

def subract_one_lable():
    ground_truth_path = 'imageNet_val/test_grond_truth.txt'
    output_path = 'test_grond_truth_zero.txt'

    f = open(ground_truth_path, 'r')
    with open(output_path, 'w') as of:
        for label in f:
            of.write(str(int(label)-1) + '\n')



def load_dat():

    pbar = tqdm(total=62000)

    # training images label path
    f = open('imageNet_val/ILSVRC2010_validation_ground_truth.txt','r')
    # validation images label path
    f_val = open('test_grond_truth_zero.txt','r')

    dat_x = []
    dat_y = []

    for lable in f:
        dat_y.append(lable)

    for f_n in os.listdir(img_path):
        img_ = os.path.join(img_path, f_n)
        img = cv2.imread(img_)
        dat_x.append(img)
        pbar.update(1)

    dat_x = np.array(dat_x)

    dat_y = np.array(dat_y)
    dat_y = dat_y.astype('int32')

    val_x = []
    val_y = []
    for f_n in os.listdir(img_val_path):

        img_ = os.path.join(img_val_path,f_n)
        img = cv2.imread(img_)
        val_x.append(img)
        pbar.update(1)

    for lable in f_val:
        val_y.append(lable)

    val_x = np.array(val_x)

    val_y = np.array(val_y)
    val_y = val_y.astype('int32')
    print('done')
    return dat_x ,val_x,dat_y ,val_y


def main():
    ttl2resize = 0
    for i in range(len(img2resize_path)):
        ttl2resize += len(os.listdir(img2resize_path[i]))

    pbar = tqdm(total=ttl2resize)
    for i in range(len(img2resize_path)):
        for f_n in os.listdir(img2resize_path[i]):
            resize_images(f_n,i)
            pbar.update(1)

if __name__ == '__main__':
    main()
    # subract_one_lable()
