import os
import copy
import torch
from pathlib import Path
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import time
from torchvision import transforms
from torch.autograd import Variable
from torchvision import models, datasets, transforms
import torch.nn.functional as F


#Train/Test split
root = "WMTdata"
data = [im for im in Path(root, "T1_tif").glob("*.tif")]; data.sort()
annotations = [im for im in Path(root, "AnnotationsTif").glob("*.tif")]; annotations.sort()

Xtrain, Xtest, ytrain, ytest = train_test_split(data, annotations, test_size=0.20, random_state=1) #Train/Test

#Dataset
class Tracts(Dataset):
    """ BrainPTM
    """
    def __init__(self, datadir, maskdir, transform=None, target_transform=None):
    
        self.imgs = datadir
        self.targets= maskdir  
        assert len(self.imgs) == len(self.targets)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,ind):
        
        image = Image.open(self.imgs[ind]).convert("RGB")
        target = Image.open(self.targets[ind])
        #print(self.imgs[ind])
        #print(self.targets[ind])
        
        if self.transform is not None:
            image = self.transform(image)
            target = self.target_transform(target)

        return image, target

#Metrics 
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

#Dataloader
train_loader= torch.utils.data.DataLoader(Tracts(
    Xtrain, ytrain,
    transform=transforms.ToTensor(),
    target_transform= transforms.ToTensor()
    ), batch_size=64, shuffle=True)

test_loader= torch.utils.data.DataLoader(Tracts(
    Xtest, ytest,
    transform=transforms.ToTensor(),
    target_transform= transforms.ToTensor()
    ), batch_size=64, shuffle=True)

model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)

cp_model = copy.deepcopy(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=255)
metrics = Evaluator(2)


#Test/Train functions

def train(epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        
        #if args.cuda:
        #data, target = data.cuda(), target.cuda() * 255.
        data, target = data.cuda(), target.cuda() 
        
        data, target = Variable(data), Variable(target).long().squeeze_(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output['out'], target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

            

def test(epoch):
    model.eval()
    metrics.reset()
    test_loss = 0.
    for data, target in test_loader:
        #if args.cuda:
        data, target = data.cuda(), target.cuda() #* 255.
        data, target = Variable(data), Variable(target).long().squeeze_(1)
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output['out'], target).item()
        pred = output['out'].cpu()
        pred = F.softmax(pred, dim=1).numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        #if demo:
        #    out = pred[0,:,:]
        #    label = target[0,:,:]
        metrics.add_batch(target, pred)

    Acc = metrics.Pixel_Accuracy()
    Acc_class = metrics.Pixel_Accuracy_Class()
    mIoU = metrics.Mean_Intersection_over_Union()
    FWIoU = metrics.Frequency_Weighted_Intersection_over_Union()

    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, len(test_loader.dataset)))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(
        Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % test_loss)
    return test_loss
  
    
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = 0.01 * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        


# Trainning
epoch =15
best_loss = None
#if load_model:
#    best_loss = test(0)
try:
    for epoch in range(1, epoch + 1):
        epoch_start_time = time.time()
        
        train(epoch)
        test_loss = test(epoch)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s '.format(
            epoch, time.time() - epoch_start_time))
        print('-' * 89)

        if best_loss is None or test_loss < best_loss:
            best_loss = test_loss
            with open("model2.pt", 'wb') as fp:
                state = model.state_dict()
                torch.save(state, fp)
        else:
            adjust_learning_rate(optimizer, 0.5, epoch)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


#Testing
#Load model 
with open("model.pt", 'rb') as fp:
    state = torch.load(fp)
    model.load_state_dict(state)
    load_model = True
    
    
#Test
epoch_start_time = time.time()
test_loss = test(1)
print('-' * 89)
print('| end of epoch {:3d} | time: {:5.2f}s '.format(
    1, time.time() - epoch_start_time))
print('-' * 89)


#Inference

demo = True
out=np.empty([128,144])
label=np.empty([128,144])
#Test
epoch_start_time = time.time()
test_loss = test(1)
print('-' * 89)
print('| end of epoch {:3d} | time: {:5.2f}s '.format(
    1, time.time() - epoch_start_time))
print('-' * 89)

demo=False

plt.subplot(121)
plt.imshow(out)
plt.subplot(122)
plt.imshow(label)
