import argparse
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms

from datasets import *
from functions import *
from nets import *
from trainer import *
from torchvision.transforms import v2

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='params', help='Path to the config file.')
parser.add_argument('--dataset_path', type=str, default='/content/data/UTK', help='dataset path')
parser.add_argument('--label_file_path', type=str, default='/content/data/UTK/age_label.npy', help='label file path')
parser.add_argument('--vgg_model_path', type=str, default='/content/models/vgg16-397923af.pth', help='pretrained age classifier')
parser.add_argument('--log_path', type=str, default='/content/logs/', help='log file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--resnet50', type=str, default='/content/resnet50.pt', help='resnet50 model path')
opts = parser.parse_args()

Download the VGG13 model if it does not exist
if not os.path.exists(opts.vgg_model_path):
    url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
    os.makedirs(os.path.dirname(opts.vgg_model_path), exist_ok=True)
    urllib.request.urlretrieve(url, opts.vgg_model_path)
    print(f"Model downloaded and saved to {opts.vgg_model_path}")

log_dir = os.path.join(opts.log_path, opts.config) + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = SummaryWriter(log_dir)

config = yaml.safe_load(open('/content/configs/' + opts.config + '.yaml', 'r'))
epochs = config['epochs']
age_min = config['age_min']
age_max = config['age_max']

batch_size = 4
img_size = (512, 512)

# Load dataset
dataset_A = MyDataSet(age_min, age_max, opts.dataset_path, opts.label_file_path, output_size=img_size, training_set=True)
dataset_B = MyDataSet(age_min, age_max, opts.dataset_path, opts.label_file_path, output_size=img_size, training_set=True)
loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
loader_B = data.DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

# Initialize trainer
trainer = Trainer(config)
trainer.initialize(opts.resnet50)
trainer.to(device)

epoch_0 = 0
if opts.multigpu:
    trainer.dataparallel()
if opts.resume:
    epoch_0 = trainer.load_checkpoint(opts.checkpoint)

# Start Training
n_iter = 0
print("Start!")
print('Reconstruction weight: ', config['w']['recon'])
print('Classification weight: ', config['w']['class'])
print('Adversarial loss weight: ', config['w']['adver'])
print('TV weight: ', config['w']['tv'])

for n_epoch in range(epoch_0, epoch_0+epochs):
    print("epoch: ", n_epoch)
    if n_epoch == 10:
        trainer.config['w']['recon'] = 0.1*trainer.config['w']['recon']
        # Load dataset at 1024 x 1024 resolution for the next 10 epochs
        batch_size = config['batch_size']
        img_size = (config['input_h'], config['input_w'])
        dataset_A = MyDataSet(age_min, age_max, opts.dataset_path, opts.label_file_path, output_size=img_size, training_set=True, transform=transform)
        dataset_B = MyDataSet(age_min, age_max, opts.dataset_path, opts.label_file_path, output_size=img_size, training_set=True, transform=transform)
        loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
        loader_B = data.DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    iter_B = iter(loader_B)
    for i, list_A in enumerate(loader_A):
        print("epoch ", n_epoch)
        print("iter ", n_iter)
        image_A, age_A = list_A
        image_B, age_B = next(iter_B)
        if age_A.size(0) != batch_size:
            break
        if age_B.size(0) != batch_size:
            iter_B = iter(loader_B)
            image_B, age_B = next(iter_B)

        image_A, age_A = image_A.to(device), age_A.to(device)
        image_B, age_B = image_B.to(device), age_B.to(device)
        trainer.update(image_A, image_B, age_A, age_B, n_iter)

        if (n_iter+1) % config['log_iter'] == 0:
            trainer.log_loss(logger, n_iter)
        if (n_iter+1) % config['image_log_iter'] == 0:
            print("start to log image!!")
            trainer.log_image(image_A, age_A, logger, n_epoch, n_iter)
        if (n_iter+1) % config['image_save_iter'] == 0:
            print("start to save image!!")
            trainer.save_image(image_A, age_A, log_dir, n_epoch, n_iter)

        n_iter += 1

    trainer.save_checkpoint(n_epoch, log_dir)
    trainer.gen_scheduler.step()
    trainer.dis_scheduler.step()

trainer.save_model(log_dir)