#!/usr/bin/env python3
"""
独立测试脚本：验证 Qwen2.5-VL 模型加载
"""
import torch
from transformers import AutoProcessor

# 尝试导入 Qwen2.5-VL 模型类
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    print("✅ Qwen2_5_VLForConditionalGeneration 导入成功")
except ImportError:
    print("❌ Qwen2_5_VLForConditionalGeneration 导入失败")
    Qwen2_5_VLForConditionalGeneration = None

# 配置
MODEL_PATH = "/home/tanxk/xiaoke/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda:6"

print(f"\n📁 模型路径: {MODEL_PATH}")
print(f"🎯 目标设备: {DEVICE}")
print(f"🔧 PyTorch 版本: {torch.__version__}")
print(f"🖥️  CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"📊 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

print("\n" + "="*50)
print("开始加载模型...")
print("="*50)

# 加载 Processor
print("\n1. 加载 Processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)
print("✅ Processor 加载成功")

# 加载模型 - 直接到 GPU
print(f"\n2. 加载模型 (直接到 {DEVICE})...")
model_kwargs = {
    "trust_remote_code": True,
    "torch_dtype": torch.float16,
    "device_map": DEVICE,
}

try:
    if Qwen2_5_VLForConditionalGeneration is not None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            **model_kwargs,
        )
        print("✅ Qwen2_5_VLForConditionalGeneration 加载成功!")
    else:
        print("⚠️ Qwen2_5_VLForConditionalGeneration 不可用，尝试 AutoModel")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            **model_kwargs,
        )
        print("✅ AutoModel 加载成功!")
    
    # 验证模型
    print(f"\n3. 模型验证:")
    print(f"   - 类型: {type(model).__name__}")
    print(f"   - 设备: {next(model.parameters()).device}")
    print(f"   - 参数数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # 检查 generate 方法
    if hasattr(model, 'generate'):
        print("   - generate 方法: ✅ 可用")
    else:
        print("   - generate 方法: ❌ 不可用")
    
    print("\n" + "="*50)
    print("🎉 VL 模型加载测试成功!")
    print("="*50)
    
except Exception as e:
    print(f"\n❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
