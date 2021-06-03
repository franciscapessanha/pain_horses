
import sys
import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler


import datasets
import hopenet
import torch.utils.model_zoo as model_zoo
import numpy as np

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
                        default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
                        default='', type=str)
    parser.add_argument('--train', dest='train', help='Path to text file containing relative paths for every example.',
                        default='', type=str)
    parser.add_argument('--val', dest='val', help='Path to text file containing relative paths for every example.',
                        default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default='', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
                        default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
                        default='', type=str)

    args = parser.parse_args()
    return args


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def do_epoch(epoch, model, loader, idx_tensor, optimizer, criterion, reg_criterion,
             softmax, alpha, n_steps, n_epochs, gpu, val=False):
    if val:
        model.eval()
        val_loss_yaw = 0
        val_loss_pitch = 0
        val_loss_roll = 0
    else:
        model.train()
    with torch.set_grad_enabled(not val):
        for i, (images, labels, cont_labels, name) in enumerate(loader):
            images = Variable(images).cuda(gpu)

            # Binned labels
            label_yaw = Variable(labels[:, 0]).cuda(gpu)
            label_pitch = Variable(labels[:, 1]).cuda(gpu)
            label_roll = Variable(labels[:, 2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:, 0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:, 1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:, 2]).cuda(gpu)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            if val:
                val_loss_yaw += loss_yaw.item()
                val_loss_pitch += loss_pitch.item()
                val_loss_roll += loss_roll.item()
            elif optimizer is not None:
                loss_seq = [loss_yaw, loss_pitch, loss_roll]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer.zero_grad()
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer.step()

            if (i + 1) % (n_steps // 10) == 0 and not val:
                print('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                      %(epoch + 1, n_epochs, i + 1, n_steps, loss_yaw.item(), loss_pitch.item(), loss_roll.item()))
    if val:
        print('Epoch [%d/%d] Validation Losses: Yaw %.4f, Pitch %.4f, Roll %.4f     Average: %.4f'
              %(epoch + 1, n_epochs, val_loss_yaw / (i + 1), val_loss_pitch / (i + 1), val_loss_roll / (i + 1),
                  (val_loss_yaw + val_loss_pitch + val_loss_roll) / (3 * (i + 1))))
        print()
        return (val_loss_yaw + val_loss_pitch + val_loss_roll) / (3 * (i + 1))
    return -1


def main():
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    print('Loading data.')

    train_trf = transforms.Compose([transforms.Resize(240),
                                    transforms.RandomCrop(224), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_trf = transforms.Compose([transforms.Resize(224),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        train_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, train_trf)
        valid_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, val_trf)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        train_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, train_trf)
        valid_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, val_trf)
    elif args.dataset == 'Synhead':
        train_dataset = datasets.Synhead(args.data_dir, args.filename_list, train_trf)
        valid_dataset = datasets.Synhead(args.data_dir, args.filename_list, val_trf)
    elif  args.dataset == 'Equines':
        train_dataset = datasets.Equines(args.data_dir, args.train, train_trf)
        valid_dataset = datasets.Equines(args.data_dir, args.val, train_trf)
    elif args.dataset == 'AFLW2000':
        train_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, train_trf)
        valid_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, val_trf)
    elif args.dataset == 'BIWI':
        train_dataset = datasets.BIWI(args.data_dir, args.filename_list, train_trf)
        valid_dataset = datasets.BIWI(args.data_dir, args.filename_list, val_trf)
    elif args.dataset == 'AFLW':
        train_dataset = datasets.AFLW(args.data_dir, args.filename_list, train_trf)
        valid_dataset = datasets.AFLW(args.data_dir, args.filename_list, val_trf)
    elif args.dataset == 'SFLW':
        train_dataset = datasets.SFLW_aug(args.data_dir, args.filename_list, train_trf)
        valid_dataset = datasets.SFLW(args.data_dir, args.filename_list, val_trf)
    elif args.dataset == 'AFLW_aug':
        train_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, train_trf)
        valid_dataset = datasets.AFLW(args.data_dir, args.filename_list, val_trf)
    elif args.dataset == 'AFW':
        train_dataset = datasets.AFW(args.data_dir, args.filename_list, train_trf)
        valid_dataset = datasets.AFW(args.data_dir, args.filename_list, val_trf)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    split = len(train_dataset) // 10
    indices = list(range(len(train_dataset)))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               num_workers=1,
                                               sampler=valid_sampler)
    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax(dim=1).cuda(gpu)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': args.lr},
                                  {'params': get_fc_params(model), 'lr': args.lr * 5}],
                                 lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 2, gamma=0.1)
    last_val_loss = -1
    print('Ready to train network.')
    k = 0
    for epoch in range(num_epochs):

        scheduler.step()
        do_epoch(epoch, model, train_loader, idx_tensor, optimizer, criterion, reg_criterion,
                 softmax, alpha, len(train_dataset) // batch_size, num_epochs, gpu)
        val_loss = do_epoch(epoch, model, valid_loader, idx_tensor, None, criterion, reg_criterion,
                            softmax, alpha, len(valid_dataset) // batch_size, num_epochs, gpu, val=True)
        if last_val_loss < 0 or val_loss < last_val_loss:
            last_val_loss = val_loss
            torch.save(model.state_dict(), 'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '_best_out_of_100.pkl')
            k = 0
        """
        elif k >= 20:
            break
        else:
            k += 1
        """
if __name__ == '__main__':
    main()
