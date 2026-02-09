import argparse
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List
from PIL import Image


def is_image_like(arr: np.ndarray) -> bool:
    """
    判断 ndarray 是否可能是图像批:
    支持形状:
      (H, W)
      (H, W, C)
      (N, H, W)
      (N, H, W, C)
      (N, C, H, W)
    通道 C 允许 1 / 3 / 4
    """
    if arr.ndim == 2:
        return True
    if arr.ndim == 3:
        h, w, c = arr.shape
        if c in (1, 3, 4):
            return True
        # 可能是 (C,H,W) 单图
        c, h, w = arr.shape
        if c in (1, 3, 4):
            return True
    if arr.ndim == 4:
        # NHWC
        if arr.shape[-1] in (1, 3, 4):
            return True
        # NCHW
        if arr.shape[1] in (1, 3, 4):
            return True
    return False


def to_hw_c(arr: np.ndarray) -> List[np.ndarray]:
    """
    将输入数组拆成单张图像 (H,W,C) 列表 (C=1/3/4) 或 (H,W).
    """
    imgs = []
    if arr.ndim == 2:  # H W
        imgs.append(arr)
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):  # H W C
            imgs.append(arr)
        elif arr.shape[0] in (1, 3, 4):  # C H W
            imgs.append(np.transpose(arr, (1, 2, 0)))
        else:
            # 无法识别
            pass
    elif arr.ndim == 4:
        # NHWC
        if arr.shape[-1] in (1, 3, 4):
            for i in range(arr.shape[0]):
                imgs.append(arr[i])
        # NCHW
        elif arr.shape[1] in (1, 3, 4):
            for i in range(arr.shape[0]):
                imgs.append(np.transpose(arr[i], (1, 2, 0)))
    return imgs


def auto_normalize(x: np.ndarray, mode: str = "auto") -> np.ndarray:
    """
    归一化到 [0,255] uint8
    mode=auto: 自动判断
      若数值范围已在 [0,1.05] -> 按 [0,1] 缩放
      若在 [-1,1] -> 映射到 [0,1]
      否则按整体 min-max
    """
    x = x.astype(np.float32)
    vmin, vmax = float(x.min()), float(x.max())
    if mode == "none":
        # 尝试直接裁剪到 [0,255] 或 [0,1]
        if vmax <= 1.05 and vmin >= 0:
            x = np.clip(x, 0, 1) * 255.0
        else:
            x = np.clip(x, 0, 255)
        return x.astype(np.uint8)

    # auto
    if vmin >= -1.01 and vmax <= 1.01:
        # 假定 [-1,1]
        x = (x + 1) / 2
        x = np.clip(x, 0, 1) * 255.0
    elif vmin >= 0 and vmax <= 1.05:
        x = np.clip(x, 0, 1) * 255.0
    else:
        if vmax - vmin < 1e-8:
            x = np.zeros_like(x)
        else:
            x = (x - vmin) / (vmax - vmin) * 255.0
    return x.astype(np.uint8)


def save_image_np(img: np.ndarray, path: Path):
    """
    img: (H,W) 或 (H,W,C) numpy
    """
    if img.ndim == 2:
        pil = Image.fromarray(img, mode="L")
    elif img.ndim == 3:
        c = img.shape[-1]
        if c == 1:
            pil = Image.fromarray(img[:, :, 0], mode="L")
        elif c == 3:
            pil = Image.fromarray(img, mode="RGB")
        elif c == 4:
            pil = Image.fromarray(img, mode="RGBA")
        else:
            raise ValueError(f"Unsupported channel count {c}")
    else:
        raise ValueError(f"Unsupported ndim {img.ndim}")
    pil.save(str(path))


def extract_npz_images(npz_path: Path,
                       outdir: Path,
                       normalize: str = "auto",
                       max_per_array: int = -1,
                       verbose: bool = True):
    """
    解压并保存 .npz 中的所有可视化数组。
    normalize: auto / none
    max_per_array: >0 时限制每个数组最多导出多少张
    """
    outdir.mkdir(parents=True, exist_ok=True)
    data = np.load(str(npz_path), allow_pickle=False)
    keys = list(data.keys())
    if verbose:
        print(f"[INFO] Loaded npz: {npz_path}, keys={keys}")

    total_saved = 0
    report = []

    for k in keys:
        arr = data[k]
        if not isinstance(arr, np.ndarray):
            if verbose:
                print(f"[SKIP] key={k} 非 ndarray: {type(arr)}")
            continue
        if not is_image_like(arr):
            if verbose:
                print(f"[SKIP] key={k} shape={arr.shape} 不像图像")
            continue
        imgs = to_hw_c(arr)
        if len(imgs) == 0:
            if verbose:
                print(f"[SKIP] key={k} shape={arr.shape} 解析失败")
            continue

        limit = len(imgs) if max_per_array <= 0 else min(len(imgs), max_per_array)
        for i in range(limit):
            im = imgs[i]
            im_u8 = auto_normalize(im, normalize)
            save_name = f"{k}_{i:05d}.png"
            save_image_np(im_u8, outdir / save_name)
        total_saved += limit
        report.append((k, arr.shape, limit))

    if verbose:
        print(f"[DONE] 共保存 {total_saved} 张图像到: {outdir}")
        for k, shape, cnt in report:
            print(f"  key={k:20s} shape={shape} -> saved {cnt}")


def build_argparser():
    ap = argparse.ArgumentParser(description="解压 .npz 并导出图像")
    ap.add_argument("--input", "-i", type=str, required=True, help=".npz 文件路径")
    ap.add_argument("--outdir", "-o", type=str, required=True, help="输出图像目录")
    ap.add_argument("--normalize", type=str, default="auto", choices=["auto", "none"],
                    help="像素归一化策略: auto / none")
    ap.add_argument("--max-per-array", type=int, default=-1,
                    help="每个数组最多导出多少张(默认 -1 不限制)")
    ap.add_argument("--silent", action="store_true", help="不打印详细日志")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    npz_path = Path(args.input)
    outdir = Path(args.outdir)
    extract_npz_images(
        npz_path=npz_path,
        outdir=outdir,
        normalize=args.normalize,
        max_per_array=args.max_per_array,
        verbose=not args.silent
    )


if __name__ == "__main__":
    main()