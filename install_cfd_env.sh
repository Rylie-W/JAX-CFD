#!/bin/bash
# 1080Ti GPU环境安装脚本 - 分步解决依赖冲突
set -e

echo "🚀 开始安装JAX-CFD环境 (1080Ti优化)"
echo "========================================"

# 1. 创建基础Python环境
echo "📦 步骤1: 创建Python 3.8环境"
conda create -n cfd python=3.8 -y
echo "✅ Python环境创建完成"

# 2. 激活环境
echo "🔧 步骤2: 激活环境"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cfd

# 3. 更新pip
echo "⬆️  步骤3: 更新pip"
pip install --upgrade pip setuptools wheel

# 4. 检查CUDA版本
echo "🔍 步骤4: 检查CUDA环境"
nvidia-smi
echo "请确认您的CUDA版本是11.x"

# 5. 安装JAX (1080Ti兼容版本)
echo "🎯 步骤5: 安装JAX GPU版本"
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 6. 验证JAX GPU安装
echo "✅ 步骤6: 验证JAX GPU安装"
python -c "import jax; print('JAX devices:', jax.devices()); print('JAX version:', jax.__version__)"

# 7. 安装核心科学计算库
echo "📊 步骤7: 安装科学计算库"
pip install numpy scipy matplotlib pandas seaborn h5py

# 8. 安装JAX生态系统
echo "🔬 步骤8: 安装JAX生态系统"
pip install flax optax chex dm-haiku dm-tree

# 9. 安装深度学习框架
echo "🧠 步骤9: 安装深度学习框架"
pip install tensorflow torch

# 10. 安装其他工具库
echo "🛠️ 步骤10: 安装工具库"
pip install tqdm pyyaml xarray imageio scikit-image networkx cloudpickle
pip install gin-config absl-py attrs dill jupyter ipython

# 11. 最终验证
echo "🔍 步骤11: 最终验证"
python -c "
import jax
import numpy as np
import tensorflow as tf
print('=== 环境验证 ===')
print(f'JAX版本: {jax.__version__}')
print(f'JAX设备: {jax.devices()}')
print(f'NumPy版本: {np.__version__}')
print(f'TensorFlow版本: {tf.__version__}')
print('✅ 所有依赖安装成功!')
"

echo ""
echo "🎉 安装完成!"
echo "================"
echo "激活环境: conda activate cfd"
echo "测试GPU: python -c \"import jax; print(jax.devices())\""
echo ""
echo "⚠️ 注意事项:"
echo "1. 确保您的代码中移除了CPU强制设置"
echo "2. 如果遇到内存问题，设置: export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8"
echo "3. 运行前检查: nvidia-smi 确认GPU可用"
