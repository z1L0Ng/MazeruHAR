import torch
from model_pytorch import HART  # 改成你的模型文件和类名
import os

# 配置
save_dir = "./exported_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "hart_model_scripted.pt")

# ====== 关键修改！传入正确的参数 ======
input_shape = (128, 6)   # 改成你的数据实际 shape
activity_count = 6       # 改成你的分类数
model = HART(input_shape=input_shape, activity_count=activity_count)
# =====================================

model.eval()

# （可选）如果你有训练好的权重，加载进来
# model.load_state_dict(torch.load('your_trained_model.pth'))

# 用 torch.jit.script 导出
scripted_model = torch.jit.script(model)

# 保存成 .pt 文件
scripted_model.save(save_path)

print(f"✅ Scripted model saved at {save_path}")

# ========== 验证一遍推理 ==========

# 构造一个 dummy 输入，注意改成你的输入 shape！
# 注意：dummy_input 也应该是 (batch_size, sequence_len, channels)
dummy_input = torch.randn(1, 128, 6)

# 推理
with torch.no_grad():
    output = scripted_model(dummy_input)

print(f"✅ Inference done, output shape: {output.shape}")

# 简单检查输出是否合理（比如分类数目对不对）
print(f"Sample output (first few logits):\n{output[0][:5]}")