import h5py
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from Datasets.Preprocessing import prepare_dataset

# TODO: Download the images: http://bcsiriuschen.github.io/CARC/
# ! tar -xzvf drive/MyDrive/Datasets/CACD2000.tar.gz
# TODO: Path to metadata
PATH='/content/drive/MyDrive/Datasets/celebrity2000.mat'
# TODO: Path to images
IMAGES='./CACD2000/'

def read_data(n=200):
    data=h5py.File(PATH,'r')
    d=data['celebrityImageData']
    files,ages=d['name'],d['age']
    rand=np.random.choice(range(files.shape[1]),size=n,replace=False)
    files= [  "".join([chr(item) for item in data[files[0][t]][:]])  for t in rand]
    ages= [ages[0][t]  for t in rand]
    return files,ages

files,ages=read_data()

# zip images to labels
imgs=[]
for i in range(len(files)):
    img=cv.imread(IMAGES+files[i],cv.IMREAD_UNCHANGED)
    img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgs.append(img)

n=5
fig, axs = plt.subplots(n,n,figsize=(2*n,3*n))
imgs_to_plot=np.random.choice(range(len(imgs)),size=n**2,replace=False)
for i in range(n):
    for j in range(n):
        indx=imgs_to_plot[i*n+j]
        axs[i,j].imshow(imgs[indx])
        axs[i,j].set_title(f'age: {ages[indx]}')
plt.show()

coral_dataset=prepare_dataset(imgs,ages)