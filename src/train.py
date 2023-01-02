import os

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import hydra
from models.trainer.image_segmentation import ImageSegmentation
from models.common.model_enum import NetType
from utils.dataset import Dataset
from torch.utils.data import DataLoader
# from utils.parse import get_parser


import albumentations as A

wandb_logger = WandbLogger(name='Single Object Segmentation',project='Background Removement')


def load_dataset(input_size, image_path, mask_path, augmentation=False):
    image_transform = None
    mask_transform = A.Compose([A.Resize(width=input_size, height=input_size)])
    if augmentation:
        mask_transform = A.Compose([
            A.Resize(width=input_size, height=input_size),
            A.RandomCrop(width=1024, height=1024),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.8),
            A.RandomRotate90(p=0.8),
            A.OneOf([
                A.ElasticTransform(alpha=50, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            ], p=0.8)]
        )

        image_transform = A.Compose([
            A.CLAHE(p=0.8),
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(p=0.8)]
        )

    dataset = Dataset(image_path=image_path, mask_path=mask_path, image_transform=image_transform,
                      mask_transform=mask_transform)
    return dataset


def load_model(model_type, model_path, gtnet_path=None):
    gtnet = None
    if gtnet_path:
        gtnet = ImageSegmentation(model_type=NetType.GTNET)
        state_dict = torch.load(gtnet_path)['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'gtnet' in key:
                key = key.replace('gtnet.','')
                new_state_dict[key] = value
        gtnet.model.load_state_dict(new_state_dict)
        print('-'*10)
        print('GTNet loaded')

    model = ImageSegmentation(model_type=model_type, gtnet=gtnet)
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path)['state_dict']
        new_state_dict = {}
        for key , value in state_dict.items():
            if 'u2net' in key:
                key = key.replace('u2net.','')
                new_state_dict[key] = value
        model.model.load_state_dict(new_state_dict)
        print('DISNet loaded')
        print('-'*10)

    return model


def get_trainer(save_model_path, batch_size, device, min_epoch, max_epoch):

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=3, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=save_model_path,
        filename="isnet-{epoch:02d}-{val_loss:.2f}-" + f"batch_size={str(batch_size)}",
        save_top_k=7,
        mode="min"
    )

    trainer = pl.Trainer(logger=wandb_logger,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         #                      devices=3, accelerator="gpu", strategy="ddp",
                         devices=2, accelerator='gpu', strategy="ddp",
                         min_epochs=min_epoch,
                         max_epochs=max_epoch,
                         profiler='simple')
    return trainer


@hydra.main(config_path="conf", config_name="default_config", version_base='1.2')
def main(cfg):
    # args = get_parser()
    model_type = NetType.DISNET
    model_path = cfg.disnet_path
    gtnet_path = cfg.gtnet_path
    net_type = cfg.net_type.upper()
    save_model_path = os.path.join('saved_models', net_type + '_crop_img')
    os.makedirs(save_model_path, exist_ok=True)
    batch_size = cfg.batch
    input_size = cfg.input_size

    # --------- Selecting Net ---------
    if net_type.__eq__(NetType.DISNET.name):
        model_type = NetType.DISNET
        model_path = cfg.disnet_path
    elif net_type.__eq__(NetType.GTNET.name):
        model_type = NetType.GTNET
        model_path = gtnet_path

    # --------- Train/ Validation Dataset Loader ---------
    train_image_path, train_mask_path = cfg.dataset.train.image_path, cfg.dataset.train.mask_path
    val_image_path, val_mask_path = cfg.dataset.val.image_path, cfg.dataset.val.mask_path
    train_dataset = load_dataset(input_size, train_image_path, train_mask_path, augmentation=cfg.dataset.augmentation)
    val_dataset = load_dataset(input_size, val_image_path, val_mask_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=cfg.dataset.train.num_workers)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=cfg.dataset.val.num_workers)

    print(f'train dataset: {len(train_dataset)}, validation dataset: {len(val_dataset)}')

    # --------- Model ---------
    print(net_type, model_type)
    model = load_model(model_type, model_path, gtnet_path=gtnet_path)
    # --------- Train ---------
    trainer = get_trainer(save_model_path, batch_size, cfg.device, cfg.min_epoch, cfg.max_epoch)
    trainer.fit(model, train_dataloader, validation_dataloader)


if __name__ == '__main__':
    main()
