import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
import cv2
import torch.backends.cudnn as cudnn

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP
from model import GTNET
from torch.utils.tensorboard import SummaryWriter

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier

writer = SummaryWriter()
# ------- 1. define loss function --------

bce_loss = nn.BCELoss(reduction='mean')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
#     print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def bce_loss_calc(d1, d2, d3, d4, d5, d6, labels_v):
    
#     tar = torch.zeros(1,1,320,320)
    
    d1 = F.upsample(d1, labels_v.shape[2:], mode='bilinear')
    d2 = F.upsample(d2, labels_v.shape[2:], mode='bilinear')
    d3 = F.upsample(d3, labels_v.shape[2:], mode='bilinear')
    d4 = F.upsample(d4, labels_v.shape[2:], mode='bilinear')
    d5 = F.upsample(d5, labels_v.shape[2:], mode='bilinear')
    d6 = F.upsample(d6, labels_v.shape[2:], mode='bilinear')
    
    
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
#     print("bce loss l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss



mse_loss = nn.MSELoss(reduction = "mean")

def feature_sync(d1, d2, d3, d4, d5, d6, g1, g2, g3, g4, g5, g6):
    loss1 = mse_loss(d1,g1)
    loss2 = mse_loss(d2,g2)
    loss3 = mse_loss(d3,g3)
    loss4 = mse_loss(d4,g4)
    loss5 = mse_loss(d5,g5)
    loss6 = mse_loss(d6,g6)

    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

#     print("mse loss l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss

# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

tra_image_dir = "/home/ubuntu/workspace/ywshin/construct/U-2-Net/train_data/DUTS-TR/DUTS-TR-Image/"
tra_label_dir = "/home/ubuntu/workspace/ywshin/construct/U-2-Net/train_data/DUTS-TR/DUTS-TR-Mask/"

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name +"_is_revised" + os.sep)

epoch_num = 2000
batch_size_train =64
batch_size_val = 16
train_num = 0
val_num = 0
cudnn.benchmark = True
tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    bname = os.path.basename(img_path).replace(image_ext, label_ext)
    
    tra_lbl_name_list.append(os.path.join(tra_label_dir, bname))

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

val_img_name_list = tra_img_name_list[-100:]
val_lbl_name_list = tra_lbl_name_list[-100:]

tra_lbl_name_list = tra_lbl_name_list[:-100]
tra_img_name_list = tra_img_name_list[:-100]


test_img = tra_img_name_list[-100:]
test_lbl = tra_lbl_name_list[-100:]
tra_lbl_name_list = tra_lbl_name_list[:-100]
tra_img_name_list = tra_img_name_list[:-100]



train_num = len(tra_img_name_list)
val_num = len(val_img_name_list)


print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("val images: ", len(val_img_name_list))
print("val labels: ", len(val_lbl_name_list))
print("---")

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT((320,320)),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=8)


val_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        RescaleT((320,320)),
        ToTensorLab(flag=0)]))
                                       
test_dataset = SalObjDataset(
    img_name_list=test_img,
    lbl_name_list=test_lbl,
    transform=transforms.Compose([
        RescaleT((320,320)),
        ToTensorLab(flag=0)]))
                                       
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=4, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# ------- 3. define model --------
# define the net
gt_net = GTNET(1, 1)

# gt_net.cuda()

net = U2NET(3, 1)
    
net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

opt_level = 'O1'
# net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)

ckpt = torch.load("/home/ubuntu/workspace/ywshin/construct/U-2-Net/hangyu/u2net_gt_bce_itr_7446_val_0.009734.pth")
gt_net.load_state_dict(ckpt['model'])
gt_net.cuda()
gt_net.eval()

for param in gt_net.parameters():
    param.requires_grad = False
    
# ------- 6. training process 2 --------
print("---start training...")


checkpoint = torch.load('saved_models/u2net_is_revised/last.pth')
net.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
# amp.load_state_dict(checkpoint['amp'])

ite_num = 0
running_loss = 0.0
# running_tar_loss = 0.0
ite_num4val = 0
save_frq = 1000 # save the model every 2000 iterations
val_loss_to_save = 10000
for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1
        inputs, labels = data['images'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        inputs_v, labels_v = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            g_hx1, g_hx2, g_hx3, g_hx4, g_hx5, g_hx6 = gt_net(labels_v)
        d1, d2, d3, d4, d5, d6, d0, hx1, hx2, hx3, hx4, hx5, hx6 = net(inputs_v)
        
        fs_mse_loss = feature_sync(g_hx1, g_hx2, g_hx3, g_hx4, g_hx5, g_hx6, hx1, hx2, hx3, hx4, hx5, hx6)
        
        gt_bce_loss = bce_loss_calc(d1, d2, d3, d4, d5, d6, labels_v)
    
        loss = fs_mse_loss + gt_bce_loss
        optimizer.zero_grad()
#         with amp.scale_loss(loss, optimizer) as scaled_loss:
#             scaled_loss.backward()
        loss.backward()
        optimizer.step()
        running_loss += gt_bce_loss.data.item()

        # # print statistics

            
    net.eval()
#     for param in net.parameters():
#         param.requires_grad = False
    
#     for i, data in enumerate(val_dataloader):

#         inputs, labels = data['images'], data['label']

#         inputs = inputs.type(torch.FloatTensor)
#         labels = labels.type(torch.FloatTensor)
#         inputs_v, labels_v = inputs.cuda(), labels.cuda()

#         with torch.no_grad():
#             d1, d2, d3, d4, d5, d6, d0, hx1, hx2, hx3, hx4, hx5, hx6 = net(inputs_v)
#             gt_bce_loss = bce_loss_calc(d1, d2, d3, d4, d5, d6, labels_v)

#         # # print statistics
#         val_running_loss += gt_bce_loss.data.item()

        

    for i, data in enumerate(test_dataloader):

        inputs, labels = data['images'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        inputs_v, labels_v = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d0, hx1, hx2, hx3, hx4, hx5, hx6 = net(inputs_v)
        pred = d0[:,0,:,:]
        pred = normPRED(pred)
        # convert torch tensor to numpy array
        pred = pred.squeeze()
        pred = pred.cpu().data.numpy()
        dst = (pred*255).astype(np.uint8)
        ori = cv2.cvtColor(cv2.imread(test_img[i]), cv2.COLOR_BGR2GRAY)
        sketch = 255-cv2.cvtColor(cv2.imread(test_lbl[i]), cv2.COLOR_BGR2GRAY)
        dst = cv2.resize(dst, dsize = (ori.shape[1],ori.shape[0]),  interpolation=cv2.INTER_AREA)
        out_dir = 'result/{}_220411/'.format(model_name)
#         os.mkdir(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, model_name + '_epoch{}'.format(epoch+1)), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir,model_name + '_epoch{}'.format(epoch+1), f"{str(i).zfill(4)}.png" ),cv2.hconcat([ori, sketch, dst]))

    print(f"val loss {running_loss}")
    
    
#     writer.add_scalar("Loss/train", running_loss / ite_num4val, epoch)
#     writer.add_scalar("Loss/val", val_running_loss / ite_num4val, epoch)
    
    
    checkpoint = {
    'model': net.state_dict(),
    'optimizer': optimizer.state_dict(),
#     'amp': amp.state_dict()
    }
    torch.save(checkpoint, model_dir +"last.pth")

    if val_loss_to_save>running_loss:
        val_loss_to_save = running_loss
        torch.save(checkpoint, model_dir + model_name+"_fs_epoch_%d_val_%3f.pth" % (epoch+1, running_loss))
        running_loss = 0.0
        val_running_loss = 0.0
        net.train()  # resume train
        ite_num4val = 0
    running_loss = 0.0
