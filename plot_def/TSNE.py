import torch
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

import os
import argparse

from ulit.init_seed import init_seed
from model.GMA1DCNN import GMA_1DCNN
from ulit.load_CWRU import load_cwru_sequential_no_cross_overlap

parser = argparse.ArgumentParser(description='PyTorch PN_Data Training')
parser.add_argument('--data', metavar='DIR', default=r'..\dataset\CWRU', help='path to dataset')
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
parser.add_argument('-seed', default=123, type=int)
parser.add_argument('-use_model', default='GMA_1DCNN', help='Model Name')

parser.add_argument('--save_model', default=True, type=bool)
parser.add_argument('--save_dir', default=r'.\result', type=str, help='保存根目录')
parser.add_argument('--save_acc_loss_dir', default=r'.\result', type=str)


def main():
    # 设备设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    args = parser.parse_args()
    init_seed(args.seed)

    label_map = {
        'Normal': 0, 'B007': 1, 'B014': 2, 'B021': 3,
        'IR007': 4, 'IR014': 5, 'IR021': 6,
        'OR007': 7, 'OR014': 8, 'OR021': 9
    }

    (train_data, train_labels), (val_data, val_labels), (
    test_data, test_labels) = load_cwru_sequential_no_cross_overlap(
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
        noise_targets=("train", "val", "test"),
        shuffle_within_split=True,
        rng_seed=args.seed,
        do_fft=True
    )

    test_data = test_data[:, None, :]

    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32),
                                 torch.tensor(test_labels, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, pin_memory=True, drop_last=False)

    model_path = f"../result/{args.use_model}/model_150.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    num_classes = len(label_map)
    if args.use_model == 'GMA_1DCNN':
        model = GMA_1DCNN(out_channel=num_classes).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"模型加载成功: {model_path}")


    features = []
    true_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.cpu().numpy()

            # 提取特征
            feature_dict = model.extract_features(inputs)
            fc_input = feature_dict['fc_input'].cpu().numpy()

            features.append(fc_input)
            true_labels.append(targets)

    features = np.concatenate(features, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    print(f"提取的特征形状: {features.shape}")
    print(f"真实标签形状: {true_labels.shape}")

    tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed, n_iter=1000)
    reduced_features = tsne.fit_transform(features)
    print(f"T-SNE降维后的特征形状: {reduced_features.shape}")


    plt.figure(figsize=(10, 8))

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    color_map = {label_name: colors[label_idx] for label_name, label_idx in label_map.items()}

    for label_id in label_map.keys():

        idx = true_labels == label_map[label_id]
        plt.scatter(
            reduced_features[idx, 0],
            reduced_features[idx, 1],
            c=color_map[label_id],
            s=50,
            alpha=0.7,
            label=label_id
        )


    plt.title(f"{args.use_model} T-SNE", fontsize=14)
    plt.xlabel("T-SNE Dimension 1", fontsize=12)
    plt.ylabel("T-SNE Dimension 2", fontsize=12)
    plt.grid(alpha=0.3)


    save_dir = os.path.dirname(model_path)
    os.makedirs(save_dir, exist_ok=True)
    save_path_no_legend = os.path.join(save_dir, f"{args.use_model}_tsne_no_legend.png")
    plt.savefig(save_path_no_legend, dpi=300, bbox_inches='tight')
    print(f"T-SNE图像（无图例）已保存至: {save_path_no_legend}")


    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_path_with_legend = os.path.join(save_dir, f"{args.use_model}_tsne_with_legend.png")
    plt.savefig(save_path_with_legend, dpi=300, bbox_inches='tight')
    print(f"T-SNE图像（带图例）已保存至: {save_path_with_legend}")

    plt.show()


if __name__ == '__main__':
    main()