import torch

import matplotlib.pyplot as plt


plt.switch_backend('TkAgg')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse


from ulit.init_seed import init_seed

from ulit.load_CWRU import load_cwru_sequential_no_cross_overlap
from model.GMA1DCNN import GMA_1DCNN

parser = argparse.ArgumentParser(description='PyTorch PN_Data Confusion Matrix Plot')
parser.add_argument('--data', metavar='DIR', default=r'..\dataset\CWRU', help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-gamma', default=0.1, type=float)
parser.add_argument('-stepsize', default=10, type=int)
parser.add_argument('-seed', default=123, type=int)
parser.add_argument('-use_model', default='GMA_1DCNN', help='Model name')
# 保存设置
parser.add_argument('--save_model', default=True, type=bool)
parser.add_argument('--save_acc_loss_dir', default=r'.\result', type=str)
parser.add_argument('--save_dir', default=r'..\result', type=str, help='Directory to save confusion matrix')


def plot_confusion_matrix(args, model, test_loader, label_map, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)  # dim=1：按类别维度取最大值

            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(targets.cpu().numpy())

    class_ids = list(label_map.values())
    cm = confusion_matrix(
        y_true=all_true_labels,
        y_pred=all_preds,
        labels=class_ids
    )


    class_names = list(label_map.keys())


    fig, ax = plt.subplots(figsize=(12, 10))  # 增大图尺寸，避免类别名重叠
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        cmap=plt.cm.Blues,
        values_format=".0f",
        xticks_rotation=0,
        colorbar=False
    )


    ax.set_xlabel("Predicted Label", fontsize=14, labelpad=10)
    ax.set_ylabel("True Label", fontsize=14, labelpad=10)


    ax.tick_params(axis='both', which='major', labelsize=12)


    plt.tight_layout()


    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.use_model}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi=300：高清图
    print(f"Confusion matrix saved to: {save_path}")


    plt.show()


def main():
    # 统一设备定义
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    args = parser.parse_args()
    init_seed(args.seed)  # 初始化随机种子，确保结果可复现


    label_map = {
        'Normal': 0, 'B007': 1, 'B014': 2, 'B021': 3,
        'IR007': 4, 'IR014': 5, 'IR021': 6,
        'OR007': 7, 'OR014': 8, 'OR021': 9
    }

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
        snr_db=0,
        noise_targets=("train", "val", "test"),  # 也可设 ("train",) 或 ("train","val") 或 ("all",)
        shuffle_within_split=True,
        rng_seed=args.seed,
        do_fft=True
    )

    # 调整数据形状为 (batch_size, 1, sequence_length)（1D CNN输入格式）
    test_data = test_data[:, None, :]

    # 创建测试数据集和加载器
    test_dataset = TensorDataset(
        torch.tensor(test_data, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    num_classes = len(label_map)
    if args.use_model == 'GMA_1DCNN':
        model = GMA_1DCNN(out_channel=num_classes).to(device)

    model_path = f"../result/{args.use_model}/model_150.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}\nPlease check the path.")


    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 设为评估模式
    print(f"Model loaded successfully from: {model_path}")

    plot_confusion_matrix(args, model, test_loader, label_map, args.save_dir)


if __name__ == '__main__':
    main()