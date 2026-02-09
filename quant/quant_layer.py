import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union



logger = logging.getLogger(__name__)
CLIPMIN = 1e-8


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


def save_tensor_hist(tensor, save_dir, name, bins=100):
    os.makedirs(save_dir, exist_ok=True)
    plt.hist(tensor.detach().cpu().numpy().flatten(), bins=bins)
    plt.title(name)
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close()


def save_tensor_txt(tensor, file_path):
    t = tensor.detach().cpu().numpy()
    t_flat = t.flatten()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 自动创建 x 目录

    np.savetxt(file_path, t_flat, fmt='%.8f')


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, always_zero: bool = False, lwc: bool = False, t_out: bool = False, svd_t: bool = False,
                auto_svd_target_energy: float = 0.95,
                auto_svd_trigger_ratio: float = 0.40,
                auto_svd_min_rank: int = 8,
                auto_svd_randomized_threshold: int = 1000000,   # 元素数超过此值用随机近似
                auto_svd_oversample: int = 8,
                auto_svd_power_iter: int = 1):
        super(UniformAffineQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.leaf_param = leaf_param
        self.scale_method = scale_method
        self.running_stat = False
        self.always_zero = always_zero

        if self.leaf_param:
            self.x_min, self.x_max = None, None

        self.lwc = lwc
        if self.lwc:
            self.upbound_factor, self.lowbound_factor, self.x_max, self.x_min = None, None, None, None
        self.sigmoid = nn.Sigmoid()

        self.t_out = t_out
        self.t_use = False
        self.x_outlier = None

        # SVD
        self.svd_t = svd_t
        # 是否低秩的判定是否完成
        self.auto_svd_decided = False
        self.svd_use = False
        # 前r个奇异值占所有方向能量的百分比
        self.auto_svd_target_energy = auto_svd_target_energy
        # 触发低秩分解的比例阈值
        self.auto_svd_trigger_ratio = auto_svd_trigger_ratio
        # 最小秩
        self.auto_svd_min_rank = auto_svd_min_rank
        # 矩阵元素超过该阈值，使用 随机化SVD
        self.auto_svd_randomized_threshold = auto_svd_randomized_threshold
        self.auto_svd_oversample = auto_svd_oversample
        self.auto_svd_power_iter = auto_svd_power_iter



        # SVD 缓存
        self.register_buffer("svd_U", None, persistent=True)
        self.register_buffer("svd_S", None, persistent=True)
        self.register_buffer("svd_VT", None, persistent=True)

        # 绘制图
        self.debug_plot_pca_ranges = getattr(self, "debug_plot_pca_ranges", True)
        self.debug_pca_topk = getattr(self, "debug_pca_topk", 64)
        self.debug_dir = getattr(self, "debug_dir", "./svd_debug")
        self.debug_layer_name = getattr(self, "debug_layer_name", "layer")

    def __repr__(self):
        s = super(UniformAffineQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     key = prefix + "x_outlier"
    #     if key in state_dict:
    #
    #         tensor = state_dict[key]
    #         if hasattr(self, "x_outlier") and "x_outlier" not in self._buffers:
    #             delattr(self, "x_outlier")
    #         if "x_outlier" in self._buffers:
    #             self._buffers["x_outlier"] = tensor.clone()
    #         else:
    #             self.register_buffer("x_outlier", tensor.clone(), persistent=True)
    #         state_dict.pop(key)
    #     super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
    #                                   missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point, self.x_outlier = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
            else:
                self.delta, self.zero_point, self.x_outlier = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        if self.running_stat:
            self.act_momentum_update(x)

        x_clone = x

        if self.svd_use:
            low_rank = (self.svd_U * self.svd_S.unsqueeze(0)) @ self.svd_VT
            low_rank_full = low_rank.view_as(x_clone)
            # 残差
            x_clone = x_clone - low_rank_full
            del low_rank
        elif self.t_use:
            xo = self.x_outlier.coalesce()
            idx = tuple(xo.indices())
            vals = xo.values()
            x_clone[idx] -= vals

        x_int = round_ste(x_clone / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        if self.svd_use:
            x_dequant = x_dequant + low_rank_full
        elif self.t_use:
            xo = self.x_outlier.coalesce()
            idx = tuple(xo.indices())
            vals = xo.values()
            x_dequant[idx] = vals

        return x_dequant

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert (self.inited)
        assert (self.leaf_param)

        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        if self.sym:
            delta = torch.max(self.x_min.abs(), self.x_max.abs()) / self.n_levels
        else:
            delta = (self.x_max - self.x_min) / (self.n_levels - 1) if not self.always_zero \
                else self.x_max / (self.n_levels - 1)

        delta = torch.clamp(delta, min=1e-8)
        if not self.sym:
            self.zero_point = (-self.x_min / delta).round() if not (self.sym or self.always_zero) else 0
        self.delta = torch.nn.Parameter(delta)

    def _randomized_top_singular_values(self, M: torch.Tensor, rank: int, oversample: int, n_iter: int):
        """仅近似返回前 rank 个奇异值（降序）"""
        with torch.no_grad():
            m, n = M.shape
            k = min(rank + oversample, min(m, n))
            Omega = torch.randn(n, k, device=M.device, dtype=M.dtype)
            Y = M @ Omega
            for _ in range(n_iter):
                Y = M @ (M.t() @ Y)
            Q, _ = torch.linalg.qr(Y, mode="reduced")
            B = Q.t() @ M
            _, S, _ = torch.linalg.svd(B, full_matrices=False)
            return S[:rank]

    def _plot_pc_ranges(self, W2d: torch.Tensor, note: str = "", highlight_k: int = None):
        """
        绘制以主成分方向(索引)为x轴的绝对幅值填充图（从0到|z|最大值的蓝色覆盖）。
        幅值定义为每个PC上的投影 z_i = sigma_i * U[:, i] 的 |z_i| 的最大值。
        highlight_k: 前 highlight_k 个PC用红色加粗线，后续PC用深蓝加粗线。
        仅用于调试可视化。
        """
        if not self.debug_plot_pca_ranges:
            return
        try:
            with torch.no_grad():
                # 精确SVD（调试时启用）
                U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
                k = int(S.shape[0])
                if k <= 0:
                    return

                # 计算每个PC的绝对幅值（max |z_i|）
                Z_abs = (U[:, :k].abs() * S[:k].unsqueeze(0))
                pc_amp = Z_abs.max(dim=0).values.detach().cpu().numpy()  # shape: (k,)
                xs = np.arange(k)

                os.makedirs(self.debug_dir, exist_ok=True)

                plt.figure(figsize=(max(6, k * 0.12), 3.0))
                plt.figure(figsize=(8, 3))
                # 从0到幅值的填充
                plt.fill_between(xs, 0.0, pc_amp, color="#5DA5DA", alpha=0.55)

                # 线条标记：前 highlight_k 红色，其余深蓝
                if highlight_k is not None:
                    h = int(max(0, min(highlight_k, k)))
                    if h > 0 and "before" in note:
                        plt.plot(xs[:h], pc_amp[:h], color="#D62728", linewidth=2.2, label=f"top-{h} PCs")
                    # if h < k:
                    #     plt.plot(xs[h:], pc_amp[h:], color="#0B3D91", linewidth=2.0, label=f"rest ({k-h}) PCs")
                else:
                    # 没有高亮需求时，给出占位图例
                    plt.plot([], [], color="#5DA5DA", linewidth=2.0, label="|z| (max abs)")

                title = ""
                if note:
                    title = f"{note}"
                plt.title(title)
                plt.xlabel("principal component index", fontsize=14)
                plt.ylabel("abs amplitude on PC", fontsize=14)
                ymax = float(pc_amp.max()) if pc_amp.size > 0 else 1.0
                plt.ylim(0.0, ymax * 1.05)
                plt.legend(loc="best", fontsize=8)

                fname = f"pc_absamp_{W2d.shape[0]}x{W2d.shape[1]}_top{k}"
                if note:
                    fname += f"_{note}"
                plt.tight_layout()

                # 保存为 SVG 文件
                svg_path = os.path.join(self.debug_dir, f"{fname}.svg")
                plt.savefig(svg_path, dpi=150, format="svg")

                plt.close()
        except Exception as e:
            logger.warning(f"[plot-pc-ranges] fail: {e}")
    def _decide_svd_and_rank(self, W2d: torch.Tensor):
        """
        根据奇异值能量自动决定是否做 SVD 及选取秩
        """
        if self.auto_svd_decided:
            return
        # 低秩分支的最大尺寸
        min_dim = min(W2d.shape)

        # 矩阵极小，低秩拆分无意义
        if min_dim <= self.auto_svd_min_rank * 2:
            # 调试可视化（即便不启用低秩，仍可画PC范围图）
            self._plot_pc_ranges(W2d, note=f"{self.debug_layer_name}-before_too_small")
            self.auto_svd_decided = True
            return

        elems = W2d.numel()

        rank_upper = min_dim
        # 获取奇异值
        if elems >= self.auto_svd_randomized_threshold:
            S = self._randomized_top_singular_values(
                W2d, rank=rank_upper,
                oversample=self.auto_svd_oversample,
                n_iter=self.auto_svd_power_iter
            )
        else:
            with torch.no_grad():
                _, S_full, _ = torch.linalg.svd(W2d, full_matrices=False)
            S = S_full[:rank_upper]

        S2 = S * S
        total = S2.sum()
        cumsum = torch.cumsum(S2, dim=0)
        target_idx = torch.searchsorted(cumsum, self.auto_svd_target_energy * total).item()
        r_energy = min(target_idx + 1, S.shape[0])
        ratio = r_energy / float(min_dim)
        use = ratio < self.auto_svd_trigger_ratio
        if use:
            self.svd_use = True
            final_rank = r_energy
            final_rank = min(final_rank, min_dim)
            final_rank = max(final_rank, self.auto_svd_min_rank)
            # final_rank = 128 if final_rank > 32 else final_rank

            # 保存截断因子
            with torch.no_grad():
                U, S_all, Vh = torch.linalg.svd(W2d, full_matrices=False)
                U_r = U[:, :final_rank]
                S_r = S_all[:final_rank]
                Vh_r = Vh[:final_rank, :]

            self.svd_U = U_r.detach()
            self.svd_S = S_r.detach()
            self.svd_VT = Vh_r.detach()

            # 绘制：分解前（原 W2d），高亮前 final_rank
            self._plot_pc_ranges(W2d, note=f"{self.debug_layer_name}-before",
                                 highlight_k=final_rank)

            # 绘制：分解后（残差 W2d - U_r Σ_r V_r^T），高亮前 final_rank
            try:
                with torch.no_grad():
                    low_rank_2d = (U_r * S_r.unsqueeze(0)) @ Vh_r
                    W2d_resid = W2d - low_rank_2d
                self._plot_pc_ranges(W2d_resid, note=f"{self.debug_layer_name}-after",
                                     highlight_k=final_rank)
            except Exception as e:
                logger.warning(f"[plot after] fail: {e}")

        else:
            # 不启用低秩：也画一张 before 便于对比（无高亮）
            self._plot_pc_ranges(W2d, note=f"{self.debug_layer_name}-before_no_use")

        self.auto_svd_decided = True
    def init_bound(self, x: torch.Tensor):
        x_clone = x.clone().detach()
        x_max = x_clone.max()
        x_min = x_clone.min()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]:
            try:
                new_max = torch.quantile(x_clone.reshape(-1), pct)
                new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
            except:
                new_max = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
                new_min = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_max, new_min)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                x_max = new_max
                x_min = new_min

        return x_max, x_min

    def weight_abs_and_outlier_removed(
            self,
            x_clone: Union["torch.Tensor", np.ndarray],
            x_outlier: Optional[Union["torch.Tensor", np.ndarray]] = None,
            *,
            debug_dir: str = "svd_debug",
            layer_name: str = "layer",
            selected_rows: Optional[Sequence[int]] = None,
            max_display_rows: int = 16,
            sample_cols_step: int = 1,
            svg_fonttype_none: bool = True,
    ) -> str:
        """
        Plot boxplots of selected rows (before and optionally after outlier removal)
        and save as an SVG file.

        Selection behavior:
        - If selected_rows is provided (non-None), use it (after clamping/validation).
        - Else if x_outlier is provided, select the top `max_display_rows` rows whose
          outlier magnitude is largest (score = max_abs(outlier) across sampled columns).
        - Else (no x_outlier and no selected_rows), fall back to uniform selection up to max_display_rows.
        """
        os.makedirs(debug_dir, exist_ok=True)
        # prefer instance layer name if present
        layer_name = getattr(self, "debug_layer_name", layer_name)

        def to_numpy(x):
            if torch is not None and isinstance(x, torch.Tensor):
                if getattr(x, "is_sparse", False):
                    x = x.to_dense()
                return x.detach().cpu().numpy()
            if isinstance(x, np.ndarray):
                return x
            return np.array(x)

        W_before = to_numpy(x_clone)
        if W_before.ndim != 2:
            raise ValueError("x_clone must be a 2-D array (rows x cols)")

        rows, cols = W_before.shape

        # prepare columns index (apply column downsampling if requested)
        cols_idx = np.arange(cols)[::max(1, sample_cols_step)]

        # determine sel (selected rows)
        if selected_rows is not None:
            # validate user-supplied indices
            sel = [int(r) for r in selected_rows if 0 <= int(r) < rows]
            if len(sel) == 0:
                raise ValueError("selected_rows contains no valid indices")
            if len(sel) > max_display_rows:
                sel = sel[:max_display_rows]
        else:
            # no explicit selected_rows -> choose based on x_outlier if available,
            # otherwise uniform sampling
            if x_outlier is not None:
                W_out_tmp = to_numpy(x_outlier)
                if W_out_tmp.shape != W_before.shape:
                    raise ValueError("x_outlier must have same shape as x_clone")
                # compute per-row outlier score: maximum absolute outlier in sampled columns
                # (use sampled cols to speed up for very wide tensors)
                scores = np.max(np.abs(W_out_tmp[:, cols_idx]), axis=1)
                topk = min(max_display_rows, rows)
                # argsort descending
                sel = np.argsort(-scores)[:topk].tolist()
            else:
                # uniform selection when no outlier provided
                if rows <= max_display_rows:
                    sel = list(range(rows))
                else:
                    sel = np.linspace(0, rows - 1, num=max_display_rows, dtype=int).tolist()

        # prepare data for before boxplots
        data_before = [W_before[r, cols_idx].ravel() for r in sel]

        # If no outlier provided, draw a single before boxplot
        if x_outlier is None:
            fig_w = max(8, int(0.6 * len(sel)))
            fig, ax = plt.subplots(figsize=(fig_w, 5))
            b = ax.boxplot(
                data_before,
                patch_artist=True,
                showfliers=True,
                widths=0.6,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color="black", linewidth=0.8),
                capprops=dict(color="black", linewidth=0.8),
                flierprops=dict(marker="o", markerfacecolor="white",
                                markeredgecolor="black", markersize=4, alpha=0.8),
            )
            for patch, c in zip(b["boxes"], plt.cm.tab20.colors):
                patch.set_facecolor(c)
            xlabels = [f"Row {r}" for r in sel]
            # ensure ticks match labels: set positions explicitly (1..n)
            n = len(xlabels)
            ticks = np.arange(1, n + 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=10)
            ax.set_title(layer_name)
            ax.set_xlabel("Row Index", fontsize=14)
            ax.set_ylabel("Values")
            ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.8)
            out_path = os.path.join(debug_dir, f"{layer_name}_boxplots_before.svg")
            fig.savefig(out_path, bbox_inches="tight", format="svg")
            plt.close(fig)
            return out_path

        # With outlier: compute after and draw before/after subplots
        # If we haven't already converted W_out_tmp above, convert now (but in the branch
        # where selected_rows was provided and x_outlier is present, W_out_tmp may be undefined).
        if 'W_out_tmp' not in locals():
            W_out_tmp = to_numpy(x_outlier)
            if W_out_tmp.shape != W_before.shape:
                raise ValueError("x_outlier must have same shape as x_clone")

        W_after = W_before - W_out_tmp
        data_after = [W_after[r, cols_idx].ravel() for r in sel]

        fig_w = max(8, int(0.6 * len(sel)))
        fig, axs = plt.subplots(2, 1, figsize=(fig_w, 8), sharex=True)
        axs = axs.flatten()

        def draw_boxplot(ax, data, title, ylabel):
            b = ax.boxplot(
                data,
                whis = 2.5,
                patch_artist=True,
                showfliers=True,
                widths=0.6,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color="black", linewidth=0.8),
                capprops=dict(color="black", linewidth=0.8),
                flierprops=dict(marker="o", markerfacecolor="white",
                                markeredgecolor="black", markersize=4, alpha=0.8),
            )
            for patch, c in zip(b["boxes"], plt.cm.tab20.colors):
                patch.set_facecolor(c)
            ax.set_title(title, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.8)

        draw_boxplot(axs[0],data_before, f"{layer_name} (before)", "Values (before)")
        draw_boxplot(axs[1], data_after, f"{layer_name} (after)", "Values (after)")

        xlabels = [f"Row {r}" for r in sel]
        n = len(xlabels)
        ticks = np.arange(1, n + 1)
        axs[1].set_xticks(ticks)
        axs[1].set_xticklabels(xlabels, rotation=35, ha="right", fontsize=10)
        axs[1].set_xlabel("Row Index", fontsize=14)

        out_path = os.path.join(debug_dir, f"{layer_name}_boxplots_before_after.svg")
        fig.savefig(out_path, bbox_inches="tight", format="svg")
        plt.close(fig)
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point, x_outlier = None, None, None
        x_clone = x.clone().detach()

        if self.svd_t:
            if x_clone.dim() == 4:
                O = x_clone.shape[0]
                W2d = x_clone.view(O, -1)
            elif x_clone.dim() == 3:
                O = x_clone.shape[0]
                W2d = x_clone.view(O, -1)
            else:  # 2D
                W2d = x_clone
            if not self.auto_svd_decided:
                self._decide_svd_and_rank(W2d)
            if self.svd_use:
                low_rank = (self.svd_U * self.svd_S.unsqueeze(0)) @ self.svd_VT
                low_rank_full = low_rank.view_as(x_clone)
                # 残差
                x_clone = x_clone - low_rank_full
                del low_rank_full, low_rank
        if channel_wise:
            # save_tensor_hist(x_clone,"save_hist",name=self.debug_layer_name)
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if self.lwc:
                init_value = 8.0
                self.upbound_factor = nn.Parameter(torch.ones(n_channels) * init_value)
                self.lowbound_factor = nn.Parameter(torch.ones(n_channels) * init_value)

                # x_max和x_min记录每个通道的最大值和最小值, 固定不变
                device_x = x_clone.device
                self.x_max = torch.zeros(n_channels, device=device_x)
                self.x_min = torch.zeros(n_channels, device=device_x)

                if self.t_out:
                    # 离群点矩阵
                    x_outlier = torch.zeros_like(x_clone)

                for c in range(n_channels):

                    if len(x.shape) == 3:
                        self.x_max[c], self.x_min[c] = self.init_bound(x_clone[:, c, :])
                    else:
                        self.x_max[c], self.x_min[c] = self.init_bound(x_clone[c])

                    # 计算离群点
                    if self.t_out and self.svd_use is False:
                        if len(x.shape) == 3:
                            x_c = x_clone[:, c, :]
                        else:
                            x_c = x_clone[c]
                        mask = (x_c > self.x_max[c]) | (x_c < self.x_min[c])

                        if len(x.shape) == 3:
                            # mask = true的位置为原值，mask = false的位置为0
                            x_outlier[:, c, :] = torch.where(mask, x_c, torch.zeros_like(x_c))
                        else:
                            x_outlier[c] = torch.where(mask, x_c, torch.zeros_like(x_c))
                        self.t_use = True
                

                if (len(x_clone.shape)==2):
                    self.weight_abs_and_outlier_removed(x_clone, x_outlier=x_outlier)

                # 如果有提取离群点，则减去离群点后计算max和min的参数
                # if self.t_out and not self.svd_use:
                #     x_outlier_s = x_outlier.to_sparse()
                #     xo = x_outlier_s.coalesce()
                #     idx = tuple(xo.indices())
                #     vals = xo.values()
                #     x_clone[idx] -= vals
                #
                #     for c in range(n_channels):
                #         if len(x.shape) == 3:
                #             self.x_max[c], self.x_min[c] = self.init_bound(x_clone[:, c, :])
                #         else:
                #             self.x_max[c], self.x_min[c] = self.init_bound(x_clone[c])
                #     x_clone[idx] = vals


                device = self.x_max.device
                xmax = self.sigmoid(self.upbound_factor).to(device) * self.x_max
                xmin = self.sigmoid(self.lowbound_factor).to(device) * self.x_min

                if self.sym:
                    abs_max = torch.max(xmax.abs(), xmin.abs())
                    scale = abs_max / self.n_levels
                    delta = scale.clamp(min=CLIPMIN, max=1e4)
                    # 量化范围的中间值
                    zero_point = torch.zeros_like(delta)
                else:
                    bound = xmax - xmin
                    scale = bound / (self.n_levels - 1)
                    delta = scale.clamp(min=CLIPMIN, max=1e4)
                    zero_point = (-(xmin) / (delta)).round()

                if len(x.shape) == 4:
                    delta = delta.view(-1, 1, 1, 1)
                    zero_point = zero_point.view(-1, 1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.view(-1, 1)
                    zero_point = zero_point.view(-1, 1)
                elif len(x.shape) == 3:
                    delta = delta.view(1, -1, 1)
                    zero_point = zero_point.view(1, -1, 1)
                else:
                    raise NotImplementedError

                # if self.t_out and self.svd_use is False:
                #     # L2计算是否足够重尾
                #     flat = x_clone.view(-1)
                #     out_vals = x_outlier.view(-1)
                #     energy_ratio = (out_vals.pow(2).sum() / (flat.pow(2).sum() + 1e-12)).item()
                #
                #
                #     # MSE判断提取离群点前后的量化误差变化
                #     x_int = round_ste(x_clone / delta) + zero_point
                #     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                #     q_base = (x_quant - zero_point) * delta
                #     mse_base = torch.mean((q_base - x) ** 2)
                #
                #     # 去掉 outlier 量化主体再加回
                #     x_outlier_s = x_outlier.to_sparse()
                #     xo = x_outlier_s.coalesce()
                #     idx = tuple(xo.indices())
                #     vals = xo.values()
                #     x_clone[idx] -= vals
                #     x_int = round_ste(x_clone / delta) + zero_point
                #     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                #     q_out = (x_quant - zero_point) * delta
                #     q_out[idx] = vals
                #     mse_out = torch.mean((q_out - x) ** 2)
                #     gain_ratio = ((mse_base - mse_out) / (mse_base + 1e-12)).item()
                #     self.t_use = (0.02 < energy_ratio < 0.09) and (gain_ratio >= 0.09)
                    # if self.t_use:
                    #     print("t_out_yes")
            else:
                if len(x.shape) == 4:
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=-1)[0]
                elif len(x.shape) == 3:
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()

                # 如果是max，会加载检查点文件，就直接跳过
                if self.t_out and 'max' not in self.scale_method:
                    # 离群点矩阵
                    x_outlier = torch.zeros_like(x_clone)

                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c], _ = self.init_quantization_scale(x_clone[:, c, :], channel_wise=False)
                    else:
                        delta[c], zero_point[c], _ = self.init_quantization_scale(x_clone[c], channel_wise=False)

                    if self.t_out and 'max' not in self.scale_method:
                        if len(x.shape) == 3:
                            x_max, x_min = self.init_bound(x_clone[:, c, :])
                            x_c = x_clone[:, c, :]
                        else:
                            x_max, x_min = self.init_bound(x_clone[c])
                            x_c = x_clone[c]

                        mask = (x_c > x_max) | (x_c < x_min)

                        if len(x.shape) == 3:
                            # mask = true的位置为原值，mask = false的位置为0
                            x_outlier[:, c, :] = torch.where(mask, x_c, torch.zeros_like(x_c))
                        else:
                            x_outlier[c] = torch.where(mask, x_c, torch.zeros_like(x_c))
                if len(x.shape) == 4:
                    delta = delta.view(-1, 1, 1, 1)
                    zero_point = zero_point.view(-1, 1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.view(-1, 1)
                    zero_point = zero_point.view(-1, 1)
                elif len(x.shape) == 3:
                    delta = delta.view(1, 1, -1)
                    zero_point = zero_point.view(1, 1, -1)
                else:
                    raise NotImplementedError
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)
            else:
                """
                4.结合
                """
                x_clone = x.clone().detach()
                x_max = x_clone.max()
                x_min = x_clone.min()
                mean_val = x_clone.mean()
                std_val = x_clone.std()
                best_score = 1e+10
                for pct in [0.999, 0.9999, 0.99999]:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                    x_q = self.quantize(x_clone, new_max, new_min)
                    score = lp_loss(x_clone, x_q, p=2, reduction='all')
                    if score < best_score:
                        best_score = score
                        x_max = new_max
                        x_min = new_min

                delta = (x_max - x_min) / (2 ** self.n_bits - 1)
                zero_point = (- x_min / delta).round()


        # if self.t_use is False:
        #     x_outlier = None
        if self.t_out and x_outlier is not None and  not x_outlier.is_sparse:
            x_outlier = x_outlier.to_sparse()

        return delta, zero_point, x_outlier







    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False,name:str=None):
        super(QuantModule, self).__init__()
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight.data
        if org_module.bias is not None:
            self.bias = org_module.bias.data
        else:
            self.bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**self.weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**self.act_quant_params)


        self.weight_quantizer.debug_layer_name = name

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor, split: int = 0):
        if not self.disable_act_quant and self.use_act_quant:
            input = self.act_quantizer(input)
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_running_stat(self, running_stat: bool):
        self.act_quantizer.running_stat = running_stat
