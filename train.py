import os
from config import Config
opt = Config('training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from data_RGB import get_training_data, get_validation_data
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from SSIM import SSIM


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

######### Set Seeds ###########
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

# TensorBoard init
writer = SummaryWriter(log_dir=model_dir)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR

######### Models ###########
from models import DMSR
model_restoration=DMSR.DMSR()
model_restoration.cuda()


device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

new_lr = opt.OPTIM.LR_INITIAL


optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                        eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_best.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_L1 = nn.L1Loss()
criterion_fft = losses.fftLoss()
criterion_SSIM = SSIM()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=12,
                          drop_last=True, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=False, num_workers=12, drop_last=True,
                        pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0

early_stopping = EarlyStopping(patience=40, min_delta=0., verbose=True)

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()

        restored = model_restoration(input_)

        label_img2 = F.interpolate(target, scale_factor=0.5, mode='bilinear')
        label_img4 = F.interpolate(target, scale_factor=0.25, mode='bilinear')
        l1 = criterion_L1(restored[2], label_img4)
        l2 = criterion_L1(restored[1], label_img2)
        l3 = criterion_L1(restored[0], target)
        loss_content = l1 + l2 + l3

        label_fft1 = torch.fft.fft2(label_img4, dim=(-2, -1))
        label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

        pred_fft1 = torch.fft.fft2(restored[2], dim=(-2, -1))
        pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

        label_fft2 = torch.fft.fft2(label_img2, dim=(-2, -1))
        label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

        pred_fft2 = torch.fft.fft2(restored[1], dim=(-2, -1))
        pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

        label_fft3 = torch.fft.fft2(target, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(restored[0], dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

        f1 = criterion_L1(pred_fft1, label_fft1)
        f2 = criterion_L1(pred_fft2, label_fft2)
        f3 = criterion_L1(pred_fft3, label_fft3)
        loss_fft = f1 + f2 + f3
        loss = loss_content + 0.1 * loss_fft

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Log training loss to TensorBoard at the end of each epoch.
    writer.add_scalar('train/Loss', epoch_loss / len(train_loader), epoch)
    writer.add_scalar('train/LearningRate', scheduler.get_lr()[0], epoch)

    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)
            restored = restored[0]
            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        # Log validation PSNR to TensorBoard at the end of each epoch.
        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            print("------save best epoch!!!-----")
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

        # Invoke the early stopping strategy.
        early_stopping(psnr_val_rgb)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch)
            break

    if epoch % opt.TRAINING.VAL_SAVE_EVERY == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

# close TensorBoard
writer.close()

# Shut down automatically after training completion.
os.system("/usr/bin/shutdown")
