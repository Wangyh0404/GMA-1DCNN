

import time
import os
import argparse
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import csv

# 假设以下模块已经定义好
from ulit.init_seed import init_seed
from ulit.acc import AverageMeter
#from ulit.load_MFS import load_mfs_sequential_no_cross_overlap

from ulit.load_CWRU import load_cwru_sequential_no_cross_overlap


#模型
from model.GMA1DCNN import *

# 设置训练参数
parser = argparse.ArgumentParser(description='PyTorch PN_Data Training')
parser.add_argument('--data', metavar='DIR', default=r'.\dataset\CWRU', help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                    help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-gamma', default=0.1, type=float)
parser.add_argument('-stepsize', default=10, type=int)
parser.add_argument('-seed', default=123,type=int)
parser.add_argument('-use_model', default='GMA_1DCNN', help='GMA_1DCNN')
# save
parser.add_argument('--save_model', default=True, type=bool)
parser.add_argument('--save_dir', default=r'.\result', type=str, help='save_root')
parser.add_argument('--save_acc_loss_dir', default=r'.\result', type=str)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    init_seed(args.seed)  # 初始化随机种子参数

    # 加载和处理数据
    #CWRU数据集
    label_map = {'Normal': 0, 'B007': 1, 'B014': 2, 'B021': 3, 'IR007': 4, 'IR014': 5, 'IR021': 6, 'OR007': 7,
                 'OR014': 8, 'OR021': 9}
    labels = ['normal', 'B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021']  # 根据需要指定标签


    #MFS数据集
    # label_map = {"Ball": 0,"Comb": 1,"Inner": 2,"Outer": 3}
    # labels = ["Ball", "Comb", "Inner", "Outer"]
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_cwru_sequential_no_cross_overlap(
        datadir=args.data,
        load_type="1HP",
        label_map=label_map,
        window_size=2048,
        overlap=0.5,
        per_file_n_train=5,
        per_file_n_val=5,
        per_file_n_test=90,
        add_noise=False,
        snr_db=0,  # SNR(dB)
        noise_targets=("train", "val", "test"),
        shuffle_within_split=True,
        rng_seed=args.seed,
        do_fft=True
    )


    train_data = train_data[:, None, :]
    val_data = val_data[:, None, :]
    test_data = test_data[:, None, :]

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)


    if args.use_model == 'GMA_1DCNN':
        model = GMA_1DCNN().to(device)
    args.save_dir = os.path.join(args.save_dir, args.use_model)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_acc_loss_dir = os.path.join(args.save_dir, 'train_test_result.csv')

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = StepLR(optimizer, gamma=args.gamma, step_size=args.stepsize)

    train_acc_list, train_loss_list, val_acc_list, val_loss_list = [], [], [], []
    #CWRU
    fault_classes = ['Normal', 'B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021']  # 故障类别
    #MFS
    # fault_classes = ["Ball", "Comb", "Inner", "Outer"]
    total_start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, device)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch, device)
        train_acc_list.append(round(train_acc, 4))
        train_loss_list.append(round(train_loss, 4))
        val_acc_list.append(round(val_acc, 4))
        val_loss_list.append(round(val_loss, 4))
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nTotal training time: {total_duration:.2f} seconds.")

    test_acc, test_loss, true_labels, pred_labels = test(test_loader, model, criterion, args.epochs - 1, device)


    test_accuracy = round(accuracy_score(true_labels, pred_labels), 4)
    test_f1 = round(f1_score(true_labels, pred_labels, average='weighted'), 4)
    test_precision = round(precision_score(true_labels, pred_labels, average='weighted'), 4)
    test_recall = round(recall_score(true_labels, pred_labels, average='weighted'), 4)

    # 保存模型
    if args.save_model:
        if epoch == args.epochs - 1:
            model_name = 'model' + '_' + str(epoch + 1) + '.pth'
            torch.save(model.state_dict(), os.path.join(args.save_dir, model_name))
            cm = confusion_matrix(true_labels, pred_labels)
            np.savetxt(os.path.join(args.save_dir, 'confusion_matrix.txt'), cm, fmt='%d')

            report = classification_report(true_labels, pred_labels, target_names=fault_classes, digits=4,
                                           output_dict=False)
            print(report)


    if args.save_model:
        with open(args.save_acc_loss_dir, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss', 'test_acc', 'test_loss',
                             'test_accuracy', 'test_f1', 'test_precision', 'test_recall'])  # 写入表头
            for epoch in range(len(train_acc_list)):
                writer.writerow([epoch + 1, round(train_acc_list[epoch], 4), round(train_loss_list[epoch], 4),
                                 round(val_acc_list[epoch], 4), round(val_loss_list[epoch], 4), "", "", "", "", "", ""])
            writer.writerow([args.epochs, "", "", "", "", round(test_acc, 4), round(test_loss, 4),
                             round(test_accuracy, 4), round(test_f1, 4), round(test_precision, 4),
                             round(test_recall, 4)])
def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, device):
    losses = AverageMeter('Loss', ':.4f')
    acc = AverageMeter('Acc', ':.4f')
    epoch_start_time = time.time()
    model.train()
    for i, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), label.size(0))
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == label).sum().item() / label.size(0)
        acc.update(accuracy, label.size(0))
    lr_scheduler.step()
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.")
    print(f'Epoch [{epoch}] Train Acc: {acc.avg:.4f} Loss: {losses.avg:.4f}')
    return acc.avg, losses.avg



def validate(val_loader, model, criterion, epoch, device):
    losses = AverageMeter('Loss', ':.4f')
    acc = AverageMeter('Acc', ':.4f')
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.update(loss.item(), label.size(0))
            _, predicted = torch.max(output, 1)
            accuracy = (predicted == label).sum().item() / label.size(0)
            acc.update(accuracy, label.size(0))
    print(f'Epoch [{epoch}] Val Acc: {acc.avg:.4f} Loss: {losses.avg:.4f}')
    return acc.avg, losses.avg


def test(test_loader, model, criterion, epoch, device):
    losses = AverageMeter('Loss', ':.4f')
    acc = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.update(loss.item(), label.size(0))
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            accuracy = (predicted == label).sum().item() / label.size(0)
            acc.update(accuracy, label.size(0))
    print(f'Epoch [{epoch}] Test Acc: {acc.avg:.4f} Loss: {losses.avg:.4f}')
    return acc.avg, losses.avg, all_labels, all_preds


if __name__ == '__main__':
    main()


