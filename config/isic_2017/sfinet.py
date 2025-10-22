from torch.utils.data import DataLoader
from seg.losses import *
from seg.datasets.isic_2017_dataset import *
from tools.utils import Lookahead
from tools.utils import process_model_params


# training hparam
max_epoch = 105
ignore_index = 255
train_batch_size = 16
val_batch_size = 1
lr = 1e-3
weight_decay = 0.0025
backbone_lr = 1e-3
backbone_weight_decay = 0.0025
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "sfinet"
weights_path = "/home/cj/codes/2d_m/fre/log/isic2017/{}".format(weights_name)
test_weights_name = weights_name
log_name = "/home/cj/codes/2d_m/fre/log/isic2017/{}".format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = [0]
pretrained_ckpt_path = None
resume_ckpt_path = None

#  define the network
from seg.models.sfinet import SFINet
net = SFINet(num_classes=2)

# define the loss
loss = JointLoss(\
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),\
    DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False

# define the dataloader
train_dataset = ISIC2017Dataset(mode='train',transform=train_aug)
val_dataset = ISIC2017Dataset(mode='val',transform=val_aug)
test_dataset = ISIC2017Dataset(mode='test',transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=32,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=16,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)


test_loader = DataLoader(dataset=test_dataset,
                        batch_size=val_batch_size,
                        num_workers=16,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)