import h5py
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import dlib
import os
from PIL import Image

_detector = dlib.get_frontal_face_detector()

def _detect_face(img):
    detected = _detector(img, 1)
    return detected[0] if len(detected)==1 else None

def _center_crop_face(img, face):
    left,right,bottom,top=face.left(),face.right(),face.bottom(),face.top()
    left=max(left,0)
    right=min(right,img.shape[1])
    top=max(top,0)
    bottom=min(bottom,img.shape[0])

    width = right - left
    height = bottom - top
    tol = 15
    up_down = 5
    diff = height-width

    if diff > 0:
            tmp = img[(top-tol-up_down):(bottom+tol-up_down),
                        (left-tol-diff//2):(right+tol+diff//2), :]
    else:
            tmp = img[(top-tol-diff//2-up_down):(bottom+tol+diff//2-up_down),
                        (left-tol):(right+tol),:]
    return tmp

def _resize_face(face, size=(152, 152)):
    return  np.array(Image.fromarray(np.uint8(face)).resize(size, Image.ANTIALIAS))

def prepare_dataset(imgs,ages):
    dataset=[]
    for i in range(len(imgs)):
        img=imgs[i]
        try:
            # Detect one face on each image.
            face=_detect_face(img)
            if face is None:
                print('face not detected:',i)
                continue

            # Center and crop the image
            face=_center_crop_face(img, face)

            # resize the image
            face=_resize_face(face)


            dataset.append((face,ages[i]))
        except Exception:
            print('error at:',i)
    print(len(dataset),'images were successfully processed!')
    return dataset