'''
Our code is partially adapted from RedNet (https://github.com/JinDongJiang/RedNet)
'''
import argparse
import os
import time
import torch

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn

# from tensorboardX import SummaryWriter

import CANet_models
import CANet_data_nyuv2 as ACNet_data
from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
from utils.utils import MaskedMSELoss
from torch.optim.lr_scheduler import LambdaLR
from dataloaders.kitti_loader import KittiDepth

# nyuv2_frq = [0.04636878, 0.10907704, 0.152566  , 0.28470833, 0.29572534,
#        0.42489686, 0.49606689, 0.49985867, 0.45401091, 0.52183679,
#        0.50204292, 0.74834397, 0.6397011 , 1.00739467, 0.80728748,
#        1.01140891, 1.09866549, 1.25703345, 0.9408835 , 1.56565388,
#        1.19434108, 0.69079067, 1.86669642, 1.908     , 1.80942453,
#        2.72492965, 3.00060817, 2.47616595, 2.44053651, 3.80659652,
#        3.31090131, 3.9340523 , 3.53262803, 4.14408881, 3.71099056,
#        4.61082739, 4.78020462, 0.44061509, 0.53504894, 0.21667766]
nyuv2_frq = []
weight_path = './data/nyuv2_40class_weight.txt'
with open(weight_path,'r') as f:
    context = f.readlines()

for x in context[1:]:
    x = x.strip().strip('\ufeff')
    nyuv2_frq.append(float(x))

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']

parser = argparse.ArgumentParser(description='Complete Depth By Kitti Dataset')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')
parser.add_argument('--data-folder',
                    default='/root/autodl-tmp/KITTI_Depth_Completion',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('--data-folder-rgb',
                    default='/root/autodl-tmp/KITTI_Depth_Completion/raw',
                    type=str,
                    metavar='PATH',
                    help='data folder rgb (default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
                    choices=input_options,
                    help='input: | '.join(input_options))

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
# image_w = 640
# image_h = 480

args.val_h = 352
args.val_w = 1216
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input

def train():
    train_data = KittiDepth('train', args)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)

    num_train = len(train_data)

    print("==> Train Loder Init Sucessed, Num_train: {}".format(num_train))

    if args.last_ckpt:
        model = CANet_models.ACNet(num_class=40, backbone='ResNet-50', pretrained=False, pcca5=True)
    else:
        model = CANet_models.ACNet(num_class=40, backbone='ResNet-50', pretrained=True, pcca5=True)
    '''
        Loss Function Init.
    '''
    depth_criterion = MaskedMSELoss()

    print("==> Loss Function Init Sucessed")

    '''
        Model Init
    '''
    model.train()
    model.to(device)

    print("==> Model Init Sucessed")


    '''
        Optimizer Init
    '''

    param_list = [
        # ... all the backbone layers use lr = args.lr
        {'params': model.conv1.parameters(), 'lr':args.lr}, {'params': model.bn1.parameters(), 'lr':args.lr},
        {'params': model.conv1_d.parameters(), 'lr': args.lr}, {'params': model.bn1_d.parameters(), 'lr': args.lr},
        {'params': model.layer1.parameters(), 'lr': args.lr}, {'params': model.layer1_d.parameters(), 'lr': args.lr},
        {'params': model.layer2.parameters(), 'lr': args.lr}, {'params': model.layer2_d.parameters(), 'lr': args.lr},
        {'params': model.layer3.parameters(), 'lr': args.lr}, {'params': model.layer3_d.parameters(), 'lr': args.lr},
        {'params': model.layer4.parameters(), 'lr': args.lr}, {'params': model.layer4_d.parameters(), 'lr': args.lr},
        {'params': model.layer1_m.parameters(), 'lr': args.lr}, {'params': model.layer2_m.parameters(), 'lr': args.lr},
        {'params': model.layer3_m.parameters(), 'lr': args.lr}, {'params': model.layer4_m.parameters(), 'lr': args.lr},
        # ... all the added layers use lr = arg.lr * 10
        {'params': model.agant0.parameters(), 'lr': args.lr*10}, {'params': model.agant1.parameters(), 'lr': args.lr*10},
        {'params': model.agant2.parameters(), 'lr': args.lr * 10}, {'params': model.agant3.parameters(), 'lr': args.lr * 10},
        {'params': model.agant4.parameters(), 'lr': args.lr * 10}, {'params': model.deconv1.parameters(), 'lr': args.lr * 10},
        {'params': model.deconv2.parameters(), 'lr': args.lr * 10}, {'params': model.deconv3.parameters(), 'lr': args.lr * 10},
        {'params': model.deconv4.parameters(), 'lr': args.lr * 10}, {'params': model.final_conv.parameters(), 'lr': args.lr * 10},
        {'params': model.final_deconv.parameters(), 'lr': args.lr * 10},{'params': model.out5_conv.parameters(), 'lr': args.lr * 10},
        {'params': model.out4_conv.parameters(), 'lr': args.lr * 10},{'params': model.out3_conv.parameters(), 'lr': args.lr * 10},
        {'params': model.out2_conv.parameters(), 'lr': args.lr * 10},{'params': model.conv_5a.parameters(), 'lr': args.lr * 10},
        {'params': model.conv_5c.parameters(), 'lr': args.lr * 10},{'params': model.pca_5.parameters(), 'lr': args.lr * 10},
        {'params': model.cca_5.parameters(), 'lr': args.lr * 10},{'params': model.pconv_5.parameters(), 'lr': args.lr * 10},
        {'params': model.cconv_5.parameters(), 'lr': args.lr * 10},{'params': model.split_conv.parameters(), 'lr': args.lr*10},
    ]

    optimizer = torch.optim.SGD(param_list, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)

    print("==> Optimizer Init Sucessed")
    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)

    lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    # writer = SummaryWriter(args.summary_dir)

    for epoch in range(int(args.start_epoch), args.epochs):

        scheduler.step(epoch)
        local_count = 0
        last_count = 0
        end_time = time.time()
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                      local_count, num_train)

        for batch_idx, sample in enumerate(train_loader):

            image = sample['image']
            depth = sample['depth']
            groundTruth = sample['gt']

            optimizer.zero_grad()
            pred = model(image, depth, args.checkpoint)

            loss = depth_criterion(pred, groundTruth)
            loss.backward()
            optimizer.step()

            local_count += image.data.shape[0]  #local_count 记录图片数量
            global_step += 1 # 记录 batch数量

            if global_step % args.print_freq == 0 or global_step == 1:

                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_log(global_step, epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                end_time = time.time()


                '''
                    writer tain info
                '''
                # for name, param in model.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')
                # grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
                # writer.add_image('image', grid_image, global_step)
                # grid_image = make_grid(depth[:3].clone().cpu().data, 3, normalize=True)
                # writer.add_image('depth', grid_image, global_step)
                #
                # # grid_image = make_grid(utils.color_label(torch.max(pred_scales[0][:3], 1)[1] + 1), 3, normalize=False, range=(0, 255))
                #
                # writer.add_image('Predicted label', grid_image, global_step)
                #
                # # grid_image = make_grid(utils.color_label(target_scales[0][:3]), 3, normalize=False, range=(0, 255))
                #
                # writer.add_image('Groundtruth label', grid_image, global_step)
                # writer.add_scalar('CrossEntropyLoss', loss.data, global_step=global_step)
                # writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=global_step)

                last_count = local_count

                save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch, 0, num_train)

    # save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
    #           0, num_train)

    print("Training completed ")

if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    train()