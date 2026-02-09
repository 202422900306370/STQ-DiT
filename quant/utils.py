import logging
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.quant_layer import QuantModule, UniformAffineQuantizer
from quant.quant_block import BaseQuantBlock
from quant.quant_model import QuantModel
from quant.adaptive_rounding import AdaRoundQuantizer
from quant.quant_layer import UniformAffineQuantizer
import numpy as np


logger = logging.getLogger(__name__)


def save_inp_oup_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                      asym: bool = False, act_quant: bool = False, batch_size: int = 32, keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :param cond: conditional generation or not
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, asym=asym, act_quant=act_quant)
    cached_batches = []
    cached_inps, cached_outs = None, None
    torch.cuda.empty_cache()

    cali_xs, cali_ts, cali_ys, cfg_scale = cali_data
    
    num = int(cali_xs.size(0) / batch_size)
    for i in range(num):
        cur_inp, cur_out = get_inp_out(
            cali_xs[i * batch_size:(i + 1) * batch_size].to(device), 
            cali_ts[i * batch_size:(i + 1) * batch_size].to(device),
            cali_ys[i * batch_size:(i + 1) * batch_size].to(device), 
            cfg_scale
        )
        if isinstance(cur_inp, tuple):
            cur_x, cur_t = cur_inp
            cached_batches.append(((cur_x.cpu(), cur_t.cpu()), cur_out.cpu()))
        else:
            cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    if isinstance(cached_batches[0][0], tuple):
        cached_inps = [
            torch.cat([x[0][0] for x in cached_batches]), 
            torch.cat([x[0][1] for x in cached_batches])
        ]
    else:
        cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    
    if isinstance(cached_inps, list):
        logger.info(f"in 1 shape: {cached_inps[0].shape}, in 2 shape: {cached_inps[1].shape}")
    else:
        logger.info(f"in shape: {cached_inps.shape}")
    logger.info(f"out shape: {cached_outs.shape}")
    torch.cuda.empty_cache()
    if keep_gpu:
        if isinstance(cached_inps, list):
            cached_inps[0] = cached_inps[0].to(device)
            cached_inps[1] = cached_inps[1].to(device)
        else:
            cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
    return cached_inps, cached_outs


def save_grad_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                   damping: float = 1., act_quant: bool = False, batch_size: int = 32,
                   keep_gpu: bool = True):
    """
    Save gradient data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param damping: damping the second-order gradient by adding some constant in the FIM diagonal
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_grad = get_grad(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, x, t, y, cfg_scale):
        self.model.eval()
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(x, t, y, cfg_scale)
            except StopForwardException:
                pass

            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    _ = self.model(x, t, y, cfg_scale)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        if len(self.data_saver.input_store) > 1 and torch.is_tensor(self.data_saver.input_store[1]):
            return (self.data_saver.input_store[0].detach(),  
                self.data_saver.input_store[1].detach()), self.data_saver.output_store.detach()
        else:
            return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()


class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(model: QuantModule, layer: Union[QuantModule, BaseQuantBlock], act_quant: bool = False):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break

def uniform_sample(lst, n):
    step = len(lst) / n
    indices = [int(i * step) for i in range(n)]
    return [lst[i] for i in indices]


def get_train_samples(args, sample_data,TDAC:bool=False):
    num_samples, num_st = args.cali_n, args.cali_st
    all_samples = num_samples * num_st
    if TDAC:
        # 加载TDAC文件
        feature_map_ori = torch.load(f'calib/feature_map_{args.num_sampling_steps}_1.pt', map_location='cpu')
        feature_map = uniform_sample(feature_map_ori, num_st)
        del feature_map_ori
        mean_mse, std_mse = 142, 142
        dense_r = mean_mse + 0.2 * std_mse
        # print(len(feature_map), indices:[0,2,...,48] -> [], step)
        # 计算每个特征图与其他特征图相似数
        dense_num = torch.zeros(len(feature_map), dtype=torch.int16)
        for i in range(len(feature_map)):
            for j in range(len(feature_map)):
                if i != j:
                    # 这个地方mse为
                    mse = torch.mean((feature_map[i] - feature_map[j]) ** 2)
                    if mse <= dense_r:
                        dense_num[i] = dense_num[i] + 1
        # 归一化dense_num相似数
        dense_num_normal = (dense_num - dense_num.min()) / (dense_num.max() - dense_num.min())

        CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
        # 计算每个特征图与其他特征图的分布差异
        Cos_dis = torch.zeros(len(feature_map))
        for i in range(len(feature_map)):
            for j in range(len(feature_map)):
                if i != j:
                    Cos_dis[i] = Cos_dis[i] + torch.sum(1 - CosineSimilarity(feature_map[i], feature_map[j]))
        # 归一化Cos_dis
        Cos_dis_normal = (Cos_dis - Cos_dis.min()) / (Cos_dis.max() - Cos_dis.min())
        l = 1.1
        # 计算每个特征图的权重
        w = dense_num_normal + l * Cos_dis_normal

        prob = w / torch.sum(w)

        all_samples = all_samples // 4

        t_num = (prob * all_samples).round().detach().clone().to(dtype=torch.int)
        t_error = all_samples - torch.sum(t_num)
        _, t_num_sort = torch.sort(t_num, descending=True)
        if t_error >= 0:
            t_num_add = t_num_sort[:t_error]
            t_num[t_num_add] += 1
        else:
            a = range(len(t_num))
            for i in reversed(a):
                if t_error == 0:
                    break
                if t_num[i] > 0:
                    t_num[i] -= 1
                    t_error = t_error + 1
                else:
                    continue
        assert torch.sum(t_num) == all_samples

        t_num = [x * 4 for x in t_num]
        logger.info(t_num)
        nsteps = len(sample_data["ts"])
        step = nsteps // num_st  # 2
        timesteps = list(range(0, nsteps, step))  # [0,2,...,48]
        xs_lst = [sample_data["xs"][i][:t_num[i // step]] for i in timesteps]
        ts_lst = [sample_data["ts"][i][:t_num[i // step]] for i in timesteps]
        ys_lst = [sample_data["y"][i][:t_num[i // step]] for i in timesteps]
        xs = torch.cat(xs_lst, dim=0)
        ts = torch.cat(ts_lst, dim=0)
        ys = torch.cat(ys_lst, dim=0)
        t_num = [0] + list(np.cumsum(t_num[:-1]))
        t_num = [t.item() if isinstance(t, torch.Tensor) else t for t in t_num]
        logger.info(t_num)
        return xs, ts, ys, t_num
    else:
        if num_st == 1:
            xs = sample_data[:num_samples]
            ts = (torch.ones(num_samples) * 800)
        else:
            # get the real number of timesteps (especially for DDIM)
            nsteps = len(sample_data["ts"])
            timesteps = list(range(0, nsteps, nsteps // num_st))
            logger.info(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')
            xs_lst = [sample_data["xs"][i][:num_samples] for i in timesteps]
            ts_lst = [sample_data["ts"][i][:num_samples] for i in timesteps]
            ys_lst = [sample_data["y"][i][:num_samples] for i in timesteps]
            xs = torch.cat(xs_lst, dim=0)
            ts = torch.cat(ts_lst, dim=0)
            ys = torch.cat(ys_lst, dim=0)
            return xs, ts, ys


def convert_adaround(model):
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                   weight_tensor=module.weight)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True:
                continue
            else:
                for name, sub_module in module.named_modules():
                    if isinstance(sub_module, QuantModule):
                        sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                weight_tensor=sub_module.weight)
        else:
            convert_adaround(module)

def _log_x_outlier_in_ckpt(ckpt):
    """
    打印 ckpt 中的 x_outlier 参数信息
    """
    found = False
    for k, v in ckpt.items():
        if "x_outlier" in k:
            found = True
            if isinstance(v, torch.Tensor):
                if v.is_sparse:
                    v_co = v.coalesce()
                    nnz = v_co._nnz()
                    print(f"[x_outlier] {k}: SPARSE shape={tuple(v_co.shape)} nnz={nnz}")
                    if nnz <= 20:
                        print(f"  indices={v_co.indices().t().tolist()}")
                        print(f"  values={v_co.values().tolist()}")
                else:
                    print(f"[x_outlier] {k}: DENSE shape={tuple(v.shape)} numel={v.numel()}")
                    if v.numel() <= 20:
                        print(f"  values={v.flatten().tolist()}")
            else:
                print(f"[x_outlier] {k}: 非 Tensor 类型: {type(v)}")
    if not found:
        print("[x_outlier] ckpt 中未发现任何 x_outlier 键")

def _log_upbound_factor_in_ckpt(ckpt, atol: float = 0.0, max_print: int = 20):
    """
    打印 ckpt 中的 upbound_factor 参数信息：
    - 输出 dtype、shape
    - 判断是否所有元素都为 8（浮点类型可用 atol 容差）
    - 若存在不为 8 的元素，打印其数量与部分索引/值样例
    """
    found = False
    for k, v in ckpt.items():
        if "upbound_factor" in k:
            found = True
            if isinstance(v, torch.Tensor):
                print(f"[upbound_factor] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                if v.numel() == 0:
                    print("  空张量")
                    continue
                if v.is_floating_point() and atol > 0:
                    ne_mask = (v - 8).abs() > atol
                else:
                    ne_mask = (v != 8)
                if ne_mask.any():
                    cnt = int(ne_mask.sum().item())
                    print(f"  非 8 元素个数: {cnt}/{v.numel()}")
                    idx = ne_mask.nonzero(as_tuple=False)
                    show = min(cnt, max_print)
                    vals = v[ne_mask].view(-1)[:show]
                    print(f"  样例索引(前{show}): {idx[:show].tolist()}")
                    print(f"  样例数值(前{show}): {vals.tolist()}")
                else:
                    print("  所有元素均为 8")
            else:
                print(f"[upbound_factor] {k}: 非 Tensor 类型: {type(v)} 值={v}")
    if not found:
        print("[upbound_factor] ckpt 中未发现任何 upbound_factor 键")
# ...existing code...

def _log_state_shapes(ckpt):
    """
    遍历并打印 state_dict 的键与对应值的形状/类型
    """
    tensor_cnt, total_elems = 0, 0
    for k, v in ckpt.items():
        if isinstance(v, torch.Tensor):
            shape = tuple(v.shape)
            print(f"[state] {k}: shape={shape} dtype={v.dtype} device={v.device}")
            tensor_cnt += 1
            total_elems += v.numel()
        else:
            print(f"[state] {k}: type={type(v)}")
    print(f"[state] tensors={tensor_cnt}, total_elems={total_elems}")


def resume_cali_model(qnn, ckpt_path, cali_data):
    print("Loading quantized model checkpoint: ", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # _log_upbound_factor_in_ckpt(ckpt=ckpt)
    _log_state_shapes(ckpt)
    print("Initializing weight quantization parameters")
    qnn.set_quant_state(True, False)
    
    cali_xs, cali_ts, cali_ys, cfg_scale = cali_data
    _ = qnn(cali_xs.cuda(), cali_ts.cuda(), cali_ys.cuda(), cfg_scale)
    # change weight quantizer from uniform to adaround
    convert_adaround(qnn)

    
    for n, m in qnn.model.named_modules():
        # if isinstance(m, AdaRoundQuantizer):
        if isinstance(m, (UniformAffineQuantizer, AdaRoundQuantizer)) and "act" not in n:
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)

    # remove act_quantizer states for now, 正好去掉了upbound_factor和lowbound_factor
    keys = [key for key in ckpt.keys() if "act" in key]
    for key in keys:
        del ckpt[key]

    qnn.load_state_dict(ckpt,strict=False)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            zero_data = m.zero_point.data
            delattr(m, "zero_point")
            m.zero_point = zero_data

            delta_data = m.delta.data
            delattr(m, "delta")
            m.delta = delta_data

    print("Initializing act quantization parameters")
    qnn.set_quant_state(True, True)
    _ = qnn(cali_xs.cuda(), cali_ts.cuda(), cali_ys.cuda(), cfg_scale)
    print("Loading quantized model checkpoint again")
    
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)

        elif isinstance(m, UniformAffineQuantizer):
            m.delta = nn.Parameter(m.delta)
            if m.zero_point is not None:
                if not torch.is_tensor(m.zero_point):
                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                else:
                    m.zero_point = nn.Parameter(m.zero_point)
                
    ckpt = torch.load(ckpt_path, map_location='cpu')
    qnn.load_state_dict(ckpt,strict=False)
    qnn.set_quant_state(weight_quant=True, act_quant=True)