import os
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from intensity_normalization.normalize import fcm

root ="BrainPTM"
annotations = [label for label in Path(root, "Annotations").glob("**/*.gz")]
data = [im for im in Path(root, "data").glob("**/*.gz")]
dataT1=[]
dataDiff =[]
dataMask= []

#Separate images into T1, Difussion and brain mask
for i in range(0,len(data)):
    if "T1" in str(data[i]):
        dataT1.append(str(data[i]))
    elif "Diffusion" in str(data[i]):
        dataDiff.append(str(data[i]))
    else:
        dataMask.append(str(data[i]))
        
dataT1.sort()
dataMask.sort()


or_left =[]
or_right=[]
cst_left=[]
cst_right=[]
        
#Stores masks of each tract in a separate list
for i in range(0,len(annotations)):
    if "OR_left" in str(annotations[i]):
        or_left.append(annotations[i])
    elif "OR_right" in str(annotations[i]):
        or_right.append(annotations[i])
        
    elif "CST_right" in str(annotations[i]):
        cst_right.append(annotations[i])
        
    elif "CST_left" in str(annotations[i]):
        cst_left.append(annotations[i])

#Tract used to perform training and testing
tract_mask=or_right

image = nib.load(dataT1[0])
mask=nib.load(dataMask[0])
normalized = fcm.fcm_normalize(image, mask)
nrmdir = os.path.join("WMTdata", "T1") # Directory to store Normalized T1 volumes

if os.path.isdir(nrmdir):
    print("The directory already exists")
else:
    os.mkdir(nrmdir)
    
    print("-" *89)
    print("Normalization of T1 volumes")
    print("-" *89)
    
    for i in range(len(dataT1)):
        assert Path(dataT1[i]).parent.name == Path(dataMask[i]).parent.name
        
        image = nib.load(dataT1[i])
        mask=nib.load(dataMask[i])
        normalized = fcm.fcm_normalize(image, mask)
        name = Path(dataT1[i]).parent.name + ".nii.gz"
        nib.save(normalized, os.path.join(nrmdir,name))
        
        if i%6 ==0:
            percent =100*i/len(dataT1)
            txt = "{percent}% normalized"
            print(txt.format(percent =np.floor(100*i/len(dataT1))))
        if i == len(dataT1) -1:
            print("100.0% normalized")

#Copy tract_mask to a separate directory
import shutil
for im in tract_mask:
    name =im.name
    case=im.parent.name
    dirp ="/home/fjmoya/FinalProject/wm-segmentation/WMTdata/Annotations"
    out = Path(dirp,case+name)
    shutil.copyfile(im, out)
