import os

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.trainer import U2NetPL, GTNetPL
from src.utils.general import tensor_to_numpy as tn

test_image_root = 'train_data/DUTS-TE/DUTS-TE-Image'

device = 'cpu'
# saved_model_path = 'saved_models/u2net_gt/20220616/isnet-epoch=42-val_loss=0.90-batch_size=8.ckpt'
# saved_model_path = 'saved_models/u2net_gt/20220617/isnet-epoch=170-val_loss=0.76-batch_size=8.ckpt'
saved_model_path = 'saved_models/u2net_gt/20220620/isnet-epoch=43-val_loss=0.20-batch_size=8.ckpt'
gtnet_model_path = 'saved_models/gtnet/gtnet-epoch=01-loss=0.07.ckpt'
gtnet = GTNetPL()

if os.path.isfile(gtnet_model_path):
    gtnet.load_from_checkpoint(gtnet_model_path)
    print('gtnet model loaded')
    print()

gtnet.to(device)
model = U2NetPL(gtnet)
# model.load_gtnet(gtnet)
if os.path.isfile(saved_model_path):
    a = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(a['state_dict'])
    print('model loaded successfully')
    print()

model.to(device)

model.eval()
# workspace/sungyong/data/DUTS/test_data/test_images
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

output_masks = []
# item = '/home/ubuntu/workspace/sungyong/data/Imgs/'
item = '/home/ubuntu/workspace/sungyong/data/P3M-10k/validation/P3M-500-P/blurred_image/'

from glob import glob

files = glob(os.path.join(item, '*.jpg'))

for idx, item in enumerate(files):
    if idx > 100:
        break

    image = Image.open(item)
    ori_image = image.copy()
    image = transform(image)
    image = image.float()
    image = image.to(device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    output = outputs[0]
    # output = F.upsample(output, mask.shape[2:], mode='bilinear')
    import numpy as np


    def tensor_to_numpy(input):
        input = tn(input)
        input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        return cv2.resize(input, (1024, 1024))


    result = np.hstack([ori_image.resize((1024, 1024)), tensor_to_numpy(output), tensor_to_numpy(outputs[1]),
                        tensor_to_numpy(outputs[2]), tensor_to_numpy(outputs[3]), tensor_to_numpy(outputs[4]),
                        tensor_to_numpy(outputs[5])])

    cv2.imwrite(f'inference_test/{os.path.basename(saved_model_path)}_{str(idx).zfill(4)}.png', result)

print('saving output masks..')

# print(output_mask)

print('inference finished')