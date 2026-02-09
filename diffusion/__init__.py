# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd_original
from . import gaussian_diffusion_calib as gd_calib
from .respace import space_timesteps
from .respace import SpacedDiffusion as SpacedDiffusion_original
from .respace_calib import SpacedDiffusion as SpacedDiffusion_calib


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    calib=False
):
    # 启用校准机制的diffusion模型
    if calib:
        gd = gd_calib
        SpacedDiffusion = SpacedDiffusion_calib
    # 原始的扩散模型
    else:
        gd = gd_original
        SpacedDiffusion = SpacedDiffusion_original
    # betas,噪声调度,扩散模型中每一步添加噪声的强度
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    # sapce_timesteps将1000步的扩散模型压缩为更少步数的工具函数
    return SpacedDiffusion(
        # 扩散步数
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        # 每一步添加的噪声
        betas=betas,
        # 模型预测 EPSILON是预测噪声；START_X是预测原始图像
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        # 方差类型
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            # 学习方差范围
            else gd.ModelVarType.LEARNED_RANGE
        ),
        # 损失函数
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )
