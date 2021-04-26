import imageio
import matplotlib.pyplot as plt
import cv2
import os
import imgaug.augmenters as iaa
import enum 
from PIL import Image
import numpy as np
from scipy import ndimage  

class Extensions(enum.Enum):
    Jpg1 = ".JPG"
    Jpg2 = ".jpg"
    Png  = ".PNG"
    Png2 = ".png"
    Jpeg = ".jpeg"

def OpenFolder(filePath):
    if os.path.exists(filePath) == False:
        os.makedirs(filePath)
    

def addImg():
    img_path_list = []
    files = os.listdir()
    
    for f in files:
        for e in Extensions:
            if f.endswith(e.value):
                img_path_list.append(f)
    
    return img_path_list    

def readImg(img):
    
    image = np.asarray(Image.open(img))
    return image
    
def writeImg(path, writeImage):
    
    OpenFolder(path)
    for i in range(0,len(addImg()),1):
        resizedImg = cv2.resize(writeImage, (200,200))  
        imageio.imwrite(path+"/"+path+str(i)+".png",resizedImg)
        
    
def Segmentation(image):
    path = "segmentation"
    
    aug = iaa.Superpixels(p_replace=0.2, n_segments=32)
    segmentImg = aug(image = image)  
    
    writeImg(path,segmentImg)

def spesificAngle(image):
    path = "spesicifAngle"
    
    x = np.random.randint(15, 75)
    rotated = ndimage.rotate(image, x, axes = (0,1), reshape = True)
    
    writeImg(path,rotated)

def bluring(image):
    path = "bluring"
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    blurImg = cv2.blur(img, ksize = (6,6))
    
    writeImg(path,blurImg)
    
def ortpoel(image):
    path = "ortpoel"
    
    aug = iaa.AveragePooling(5)
    segmentImage = aug(image = image) 
    
    writeImg(path,segmentImage)

def noise(image):
    path = "noise"

    aug = iaa.AdditiveGaussianNoise(scale=0.2*255)
    noiseImage = aug(image = image) 
    
    writeImg(path,noiseImage)
    
def alphaBlend(image):
    path = "alphablend"
    
    aug = iaa.BlendAlpha(
    factor=(0.2, 0.8),
    foreground=iaa.Affine(rotate=(-20, 20)),
    per_channel=True
    )
    alphaBlendImage = aug(image = image) 
    
    writeImg(path,alphaBlendImage)

def Lambda(image):    
    path = "lambda"
    
    def img_func(images, random_state, parents, hooks):
        for img in images:
            img[::4] = 0
            return images

    def keypoint_func(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    aug = iaa.Lambda(img_func, keypoint_func)
    lambdaImage = aug(image = image) 
    
    writeImg(path,lambdaImage)

def zoomBlur(image):
    path = "zoomBlur"
    
    aug = iaa.imgcorruptlike.ZoomBlur(severity=4)
    zoomBlurImage = aug(image = image) 
    
    writeImg(path,zoomBlurImage)
    
def main():
    
    AddImage = addImg()
  
    for i in AddImage:
    
        ReadImg = readImg(i)
            
        Segmentation(ReadImg)
        bluring(ReadImg)
        spesificAngle(ReadImg)
        ortpoel(ReadImg)
        noise(ReadImg)
        alphaBlend(ReadImg)
        Lambda(ReadImg)
        zoomBlur(ReadImg)
            
            
            
            
            
            
            
            
            
            
    
        
main()
     

