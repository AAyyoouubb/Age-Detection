import os

import pandas as pd

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from Datasets.Preprocessing import prepare_dataset


'''
TODO: Download images and metadata from the link below:
https://susanqq.github.io/UTKFace/

%%bash
cd /content/drive/MyDrive/Datasets/
tar -xzvf part1.tar.gz -C /content/
tar -xzvf part2.tar.gz -C /content/
tar -xzvf part3.tar.gz -C /content/
cd /content/
mv part3/* part2/
mv part1/* part2/
mv part2 utk
rm -rf part*
'''

imgs,ages=[],[]
portion=round(0.01*len(os.listdir('utk')))
for f in os.listdir('utk'):
    if len(imgs)==portion:
        break
    img=cv.imread('utk/'+f,cv.IMREAD_UNCHANGED)
    img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgs.append(img)
    ages.append(f.split('_')[0])

# Now, let's visuallize some sample images
n=5
fig, axs = plt.subplots(n,n,figsize=(2*n,3*n))
imgs_to_plot=np.random.choice(range(len(imgs)),size=n**2,replace=False)
for i in range(n):
    for j in range(n):
        indx=imgs_to_plot[i*n+j]
        axs[i,j].imshow(imgs[indx])
        axs[i,j].set_title(f'age: {ages[indx]}')
plt.show()

# Prepare the dataset
utk_dataset=prepare_dataset(imgs,ages)