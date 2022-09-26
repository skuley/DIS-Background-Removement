import cv2

from src.models.net import U2NetPL
from src.models.gt_net import GTNetPL
from src.utils.dataset import Dataset
from src.utils.general import tensor_to_numpy
import torch
import torch.nn.functional as F
from torchvision import transforms
import os

test_image_root = 'train_data/DUTS-TE/DUTS-TE-Image'
# test_image_root = 'test_data/test_images'
test_mask_root = 'train_data/DUTS-TE/DUTS-TE-Mask'

dataset = Dataset(
    root='../data',
    img_root = test_image_root,
    img_transform=transforms.Compose([
        transforms.Resize((1024,1024)),
        # transforms.RandomVerticalFlip(p=0.5)
    ]),
    
)

save_length = 1000
device = 2
saved_model_path = 'saved_models/u2net_gt/best_model/u2net_gt-epoch=92-val_loss=0.52-batch_size=64.ckpt'
gtnet_model_path = 'saved_models/gtnet/gtnet-epoch=01-loss=0.07.ckpt'
gtnet = GTNetPL()

if os.path.isfile(gtnet_model_path):
    gtnet.load_from_checkpoint(gtnet_model_path)
    print('gtnet model loaded')
    print()
    
gtnet.to(device)
model = U2NetPL()
model.load_gtnet(gtnet)
if os.path.isfile(saved_model_path):
    a = torch.load(saved_model_path)
    model.load_state_dict(a['state_dict'])
    print('model loaded successfully')
    print()
    
model.to(device)

model = model.eval()
gtnet.eval()

# save_origin_mask_path = 'images/inference_images/original_mask'
# os.makedirs(save_origin_mask_path, exist_ok=True)
mae = 0.0
output_masks = []

# print('saving original masks..')
for idx in range(len(dataset)):
    image = dataset[idx]['image']
    image = image.float()
    image = image.to(device)
    image = image.unsqueeze(0)
    # np_mask = tensor_to_numpy(mask)
    # cv2.imwrite(os.path.join(save_origin_mask_path, f'{str(idx).zfill(4)}.png'), np_mask)
#     print(images.shape)
    with torch.no_grad():
        outputs = model(image)
    output = outputs[0]
    output = F.upsample(output, image.shape[2:], mode='bilinear')
    output_masks.append(tensor_to_numpy(output))
    # mae += mae_loss(mask, output)
    
# mae_loss = (mae/save_length).detach().cpu().numpy().item()
save_output_mask_path = f'images/inference_images/{os.path.basename(saved_model_path)}_3'
os.makedirs(save_output_mask_path, exist_ok=True)

print('saving output masks..')
for idx, output_mask in enumerate(output_masks):
    cv2.imwrite(os.path.join(save_output_mask_path, f'{str(idx).zfill(4)}.png'), output_mask)
    # print(output_mask)

print('inference finished')