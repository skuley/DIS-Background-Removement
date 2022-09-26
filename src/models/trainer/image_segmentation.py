import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from models.nets import DISNet, GTNet
from models.common.model_enum import NetType
from utils.loss import feature_sync, bce_loss_calc
from utils.general import tensor_to_numpy

import torch
from torch.nn.functional import interpolate
import pytorch_lightning as pl

import cv2
import datetime


class ImageSegmentation(pl.LightningModule):
    def __init__(self, model_type=None, gtnet=None):
        super().__init__()
        self.model_type = model_type
        if self.model_type == NetType.DISNET:
            self.model = DISNet(3, 1, model_type=self.model_type)

            if gtnet:
                self.gtnet = gtnet
                # self.gtnet.freeze()
        else:
            self.model = GTNet(1, 1)
            
        self.val_loss = 100.0

    def forward(self, x):
        return self.model(x)

    def get_loss(self, image, mask):
        if self.model_type == NetType.GTNET:
            image = mask

        side_outputs, feature_maps = self.model(image)
        loss = bce_loss_calc(side_outputs, mask)

        if self.model_type == NetType.DISNET:
            with torch.no_grad():
                _, gt_feature_maps = self.gtnet(mask)
            fs_mse_loss = feature_sync(gt_feature_maps, feature_maps)
            loss += fs_mse_loss
        return loss

    def write_images(self, side_outputs, idx, image, mask, save_image_path):
        output_mask = interpolate(side_outputs[0], size=image.shape[2:], mode='bilinear')
        output_mask = tensor_to_numpy(output_mask)
        output_mask = cv2.cvtColor(output_mask, cv2.COLOR_GRAY2BGR)

        image = tensor_to_numpy(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = tensor_to_numpy(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cat_img = cv2.hconcat([image, mask, output_mask])
        save_image_path = os.path.join(save_image_path, f'epoch_{str(self.current_epoch)}')
        os.makedirs(save_image_path, exist_ok=True)
        image_path = os.path.join(save_image_path, f'{str(idx).zfill(4)}.png')
        cv2.imwrite(image_path, cat_img)

    def training_step(self, batch, batch_idx):
        image, mask = batch['image'], batch['mask']
        loss = self.get_loss(image, mask)
        loss_log = {'loss': loss}
        self.log_dict(loss_log)
        return loss_log

    def validation_step(self, batch, batch_idx):
        now = datetime.datetime.now()
        save_image_path = f'validation_images/{now.strftime("%Y%m%d")}'
        os.makedirs(save_image_path, exist_ok=True)
        loss_list = []
        for idx, (image, mask) in enumerate(zip(batch['image'], batch['mask'])):
            
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            side_outputs, _ = self.model(image)
            loss_list.append(self.get_loss(image, mask))
            if idx <= 100:
                self.write_images(side_outputs, idx, image, mask, save_image_path)
            
        val_loss = {'val_loss': sum(loss_list)}
        if val_loss['val_loss'] < self.val_loss:
            self.val_loss = val_loss['val_loss']
        self.log_dict(val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint['val_loss'] = self.val_loss

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        model_type = checkpoint['hyper_parameters']['model_type']
        check_keyword = 'disnet.'
        if model_type == NetType.GTNET:
            check_keyword = 'gtnet.'
        new_state_dict = {}

        for key, value in state_dict.items():
            if model_type in key:
                key = key.replace(model_type + '.', '')
            new_state_dict[key] = value
        checkpoint['state_dict'] = new_state_dict


if __name__ == '__main__':
    disnet = ImageSegmentation(NetType.DISNET)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    disnet.load_from_checkpoint(
        checkpoint_path='/Users/hongsung-yong/Projects/BackgroundRemovement/saved_models/best_models/disnet.ckpt',
        map_location=device, model_type=disnet.model_type)
