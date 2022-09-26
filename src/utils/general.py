import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from utils.loss import mae_loss



def tensor_to_numpy(tensor_img):
    img_np = np.array(tensor_img.cpu().detach().squeeze(0)*255, np.uint8)
    img_np = img_np.transpose(1,2,0).squeeze()
    return img_np


def model_evaluation(model, input_image, input_mask):
    with torch.no_grad():
        output_data = model(input_image)
    if len(output_data) > 1:
        output_data = output_data[0]
    output_data = F.upsample(output_data, input_mask.shape[2:], mode='bilinear')
    # print(output_data.shape, input_mask.shape)
    return mae_loss(input_mask, output_data), output_data

