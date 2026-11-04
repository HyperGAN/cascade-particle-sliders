"""Explicit execution profiles; callers own GPU admission and global flags."""
import os

import torch

CPU_PROFILE = 'cpu-fp32-v1'
CUDA_PROFILE = 'native-cuda-fp32-math-v1'


def execution_policy(options):
    profile = options.get('runtime_profile', CPU_PROFILE)
    device = options.get('device', 'cpu')
    if options.get('base_precision', 'float32') != 'float32':
        raise ValueError('Stage C execution profiles require float32')
    if profile == CPU_PROFILE:
        if device != 'cpu':
            raise ValueError('CPU profile requires device=cpu')
        return dict(profile=profile, device=device, base_precision='float32')
    if profile != CUDA_PROFILE or device != 'cuda:0':
        raise ValueError('select an explicit Stage C execution profile and its device')
    flags = dict(
        deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    )
    required = dict(deterministic=True, warn_only=False, matmul_tf32=False,
                    cudnn_tf32=False, cudnn_benchmark=False, cudnn_deterministic=True,
                    flash_sdp=False, efficient_sdp=False, cudnn_sdp=False,
                    math_sdp=True, cublas_workspace=':4096:8')
    if flags != required:
        raise ValueError('native CUDA profile requires caller-set strict deterministic/math flags')
    # Checking this policy never initializes CUDA or modifies caller settings.
    return dict(profile=profile, device=device, base_precision='float32',
                torch=str(torch.__version__), cuda=torch.version.cuda, flags=flags)
