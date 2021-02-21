import pandas as pd

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from Datasets.Preprocessing import prepare_dataset


'''
TODO: Download images and metadata from the links below
%%bash
git clone https://github.com/afad-dataset/tarball.git
cd /content/tarball/
cat AFAD-Full.tar.xz* > Full.tar.xz
rm  AFAD-Full.tar.xz*
mv Full.tar.xz AFAD-Full.tar.xz
tar -xvf AFAD-Full.tar.xz
wget https://raw.githubusercontent.com/Raschka-research-group/coral-cnn/master/datasets/afad_test.csv
wget https://raw.githubusercontent.com/Raschka-research-group/coral-cnn/master/datasets/afad_train.csv
wget https://raw.githubusercontent.com/Raschka-research-group/coral-cnn/master/datasets/afad_valid.csv
'''

# TODO: path to the images
PATH='/content/tarball/AFAD-Full/'
# TODO: path to csv files
CSV='./content/tarball'

def df(name):
    return pd.read_csv(CSV+name)
names=['afad_train.csv','afad_test.csv','afad_valid.csv']
dfs=[df(t) for t in names]
frame = pd.concat(dfs, ignore_index=True)
frame.path=frame.path.apply(lambda x: PATH+str(x))

frame=frame.sample(frac=1.)
portion=frame.shape[0]//500
imgs=[]
ages=[]
for i in range(portion):
    img=cv.imread(frame.loc[i,'path'],cv.IMREAD_UNCHANGED)
    img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgs.append(img)
    ages.append(frame.loc[i,'path'].split('/')[-3])

# Now, let's visuallize some sample images.
n=5
fig, axs = plt.subplots(n,n,figsize=(2*n,3*n))
imgs_to_plot=np.random.choice(range(len(imgs)),size=n**2,replace=False)
for i in range(n):
    for j in range(n):
        indx=imgs_to_plot[i*n+j]
        axs[i,j].imshow(imgs[indx])
        axs[i,j].set_title(f'age: {ages[indx]}')
plt.show()


# Prepare the dataset:
afad_dataset=prepare_dataset(imgs,ages)