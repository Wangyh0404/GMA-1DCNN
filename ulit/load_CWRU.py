import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional
from scipy.io import loadmat

# ====================== 频域与噪声工具 ======================
def rfft_half_drop_nyquist(wins_2d: np.ndarray) -> np.ndarray:
    spec = np.fft.rfft(wins_2d, axis=1)
    mag  = (np.abs(spec) / wins_2d.shape[1])
    return mag[:, :-1].astype(np.float32)

def add_awgn_per_window(wins_2d: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    if wins_2d.size == 0:
        return wins_2d
    sig_power = np.mean(wins_2d.astype(np.float64)**2, axis=1, keepdims=True) + 1e-12
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = sig_power / snr_linear
    noise = rng.randn(*wins_2d.shape).astype(np.float64)
    noise_power = np.mean(noise**2, axis=1, keepdims=True) + 1e-12
    noise = noise * np.sqrt(noise_var / noise_power)
    return (wins_2d.astype(np.float64) + noise).astype(np.float32)


def _parse_cwru_filename(mat_name: str) -> Tuple[str, int, int]:
    parts = mat_name.split('_')
    if len(parts) < 3:
        raise ValueError(f"[CWRU] unexpected filename format: {mat_name}")
    file_label = parts[0]
    try:
        load_id = int(parts[1])
        file_id = int(parts[2].split('.')[0])
    except Exception:
        raise ValueError(f"[CWRU] invalid load/file id in: {mat_name}")
    return file_label, load_id, file_id

def _axis_name_candidates(file_id: int):
    suffixes = ["_DE_time", "_FE_time", "_BA_time"]
    return [f"X{file_id:03d}{sfx}" for sfx in suffixes]

def _read_cwru_vector_auto_axis(mat_path: str) -> np.ndarray:
    md = loadmat(mat_path)
    base = os.path.basename(mat_path)
    _, _, file_id = _parse_cwru_filename(base)
    for var in _axis_name_candidates(file_id):
        if var in md:
            v = md[var].squeeze().astype(np.float32)
            return v.reshape(-1).astype(np.float32)
    for _, v in md.items():  # 兜底：取第一个一维数值数组
        if isinstance(v, np.ndarray):
            arr = v.squeeze()
            if arr.ndim == 1 and arr.size > 0 and arr.dtype.kind in ("f", "i", "u"):
                return arr.astype(np.float32)
    raise KeyError(f"[CWRU] no suitable axis found in: {mat_path}")

def _needs_noise(split_name: str, noise_targets: Iterable[str]) -> bool:
    targets = set(t.lower() for t in noise_targets)
    return ("all" in targets) or (split_name.lower() in targets)

def sequential_split_no_cross_overlap(starts: np.ndarray,
                                      window_size: int,
                                      step: int,
                                      n_train: int,
                                      n_val: int,
                                      n_test: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    total = len(starts)
    if total == 0:
        return np.array([], int), np.array([], int), np.array([], int)

    n_train = max(0, min(n_train, total))
    train_idx = np.arange(0, n_train, dtype=int)

    def _first_index_after_boundary(bound_end_sample: int) -> int:
        if total == 0:
            return total
        s0 = starts[0]
        k = int(np.ceil(max(0, bound_end_sample - s0) / float(step)))
        return min(k, total)

    if n_train > 0:
        last_train_start = starts[n_train - 1]
        train_last_end   = last_train_start + window_size
        k_val_start = _first_index_after_boundary(train_last_end)
    else:
        k_val_start = 0

    k_val_end = min(k_val_start + n_val, total)
    val_idx = np.arange(k_val_start, k_val_end, dtype=int)

    if val_idx.size > 0:
        last_val_start = starts[val_idx[-1]]
        val_last_end   = last_val_start + window_size
        k_test_start = _first_index_after_boundary(val_last_end)
    elif n_train > 0:
        last_train_start = starts[n_train - 1]
        train_last_end   = last_train_start + window_size
        k_test_start = _first_index_after_boundary(train_last_end)
    else:
        k_test_start = 0

    k_test_end = min(k_test_start + n_test, total)
    test_idx = np.arange(k_test_start, k_test_end, dtype=int)

    return train_idx, val_idx, test_idx

def window_and_split_one_signal(x: np.ndarray,
                                window_size: int = 2048,
                                overlap: float = 0.5,
                                n_train: int = 5,
                                n_val: int = 5,
                                n_test: int = 90,
                                # 噪声相关
                                add_noise: bool = False,
                                snr_db: float = 12.0,
                                noise_targets: Iterable[str] = ("train", "val", "test"),
                                # 特征模式
                                do_fft: bool = True,
                                rng: Optional[np.random.RandomState] = None
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = window_size
    step = max(1, int(W * (1.0 - overlap)))
    if x.size < W:
        out_dim = (W // 2) if do_fft else W
        empty = np.empty((0, out_dim), np.float32)
        return empty, empty, empty

    starts = np.arange(0, x.size - W + 1, step, dtype=int)
    tr_idx, va_idx, te_idx = sequential_split_no_cross_overlap(starts, W, step, n_train, n_val, n_test)

    # 集合内独立 RNG
    if rng is None:
        rng = np.random.RandomState(12345)
    rng_tr = np.random.RandomState(rng.randint(0, 2**31 - 1))
    rng_va = np.random.RandomState(rng.randint(0, 2**31 - 1))
    rng_te = np.random.RandomState(rng.randint(0, 2**31 - 1))

    def _collect(idx_arr: np.ndarray, split_name: str, split_rng: np.random.RandomState) -> np.ndarray:
        if idx_arr.size == 0:
            out_dim = (W // 2) if do_fft else W
            return np.empty((0, out_dim), dtype=np.float32)
        wins = np.stack([x[s:s+W] for s in starts[idx_arr]], axis=0).astype(np.float32)
        if add_noise and _needs_noise(split_name, noise_targets):
            wins = add_awgn_per_window(wins, snr_db=snr_db, rng=split_rng)
        return rfft_half_drop_nyquist(wins) if do_fft else wins

    Xtr = _collect(tr_idx, "train", rng_tr)
    Xva = _collect(va_idx, "val",   rng_va)
    Xte = _collect(te_idx, "test",  rng_te)
    return Xtr, Xva, Xte

def load_cwru_sequential_no_cross_overlap(
    datadir: str,
    load_type: str,
    label_map: Dict[str, int],
    window_size: int = 2048,
    overlap: float = 0.5,
    # 每文件采样数量（train/val/test）
    per_file_n_train: int = 5,
    per_file_n_val: int = 5,
    per_file_n_test: int = 90,
    shuffle_within_split: bool = True,
    rng_seed: int = 42,
    # 噪声控制
    add_noise: bool = False,
    snr_db: float = 12.0,
    noise_targets: Iterable[str] = ("train", "val", "test"),
    # ★ 新增：是否做 FFT
    do_fft: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    base = Path(datadir) / load_type
    if not base.exists():
        raise FileNotFoundError(f"{base} not found")

    Xtr_list, ytr_list = [], []
    Xva_list, yva_list = [], []
    Xte_list, yte_list = [], []

    top_rng = np.random.RandomState(rng_seed)

    mat_files = [f for f in sorted(os.listdir(base)) if f.lower().endswith(".mat")]
    if len(mat_files) == 0:
        print(f"[WARN] no .mat files under {base}")

    for fname in mat_files:
        try:
            file_label, _, _ = _parse_cwru_filename(fname)
        except ValueError as e:
            print(f"[WARN] {e}; skip file: {fname}")
            continue

        if file_label not in label_map:
            print(f"[WARN] skip unknown label '{file_label}' in: {fname}")
            continue
        lab = label_map[file_label]

        vec = _read_cwru_vector_auto_axis(str(base / fname))
        file_rng = np.random.RandomState(top_rng.randint(0, 2**31 - 1))

        Xtr, Xva, Xte = window_and_split_one_signal(
            vec,
            window_size=window_size,
            overlap=overlap,
            n_train=per_file_n_train,
            n_val=per_file_n_val,
            n_test=per_file_n_test,
            add_noise=add_noise,
            snr_db=snr_db,
            noise_targets=noise_targets,
            do_fft=do_fft,                 # ★ 使用开关
            rng=file_rng
        )

        if Xtr.size: Xtr_list.append(Xtr); ytr_list.append(np.full((Xtr.shape[0],), lab, dtype=np.int64))
        if Xva.size: Xva_list.append(Xva); yva_list.append(np.full((Xva.shape[0],), lab, dtype=np.int64))
        if Xte.size: Xte_list.append(Xte); yte_list.append(np.full((Xte.shape[0],), lab, dtype=np.int64))

        print(f"[CWRU OK] {load_type}/{fname} -> train={Xtr.shape[0]} val={Xva.shape[0]} test={Xte.shape[0]}")

    def _concat(Xl, yl, W=window_size, do_fft_flag=True):
        feat_dim = (W // 2) if do_fft_flag else W
        if len(Xl) == 0:
            return np.empty((0, feat_dim), np.float32), np.empty((0,), np.int64)
        X = np.concatenate(Xl, axis=0).astype(np.float32)
        y = np.concatenate(yl, axis=0).astype(np.int64)
        return X, y

    Xtr, ytr = _concat(Xtr_list, ytr_list, window_size, do_fft)
    Xva, yva = _concat(Xva_list, yva_list, window_size, do_fft)
    Xte, yte = _concat(Xte_list, yte_list, window_size, do_fft)

    if shuffle_within_split:
        rng = np.random.RandomState(rng_seed + 2025)
        def _shuffle(X, y):
            if X.size == 0:
                return X, y
            idx = rng.permutation(X.shape[0])
            return X[idx], y[idx]
        Xtr, ytr = _shuffle(Xtr, ytr)
        Xva, yva = _shuffle(Xva, yva)
        Xte, yte = _shuffle(Xte, yte)

    print(f"[CWRU SHAPE] TRAIN: {Xtr.shape}, VAL: {Xva.shape}, TEST: {Xte.shape} | "
          f"do_fft={do_fft} add_noise={add_noise}, snr_db={snr_db}, targets={tuple(noise_targets)}")
    return (Xtr, ytr), (Xva, yva), (Xte, yte)
