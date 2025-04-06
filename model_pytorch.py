import torch
import torch.nn as nn
import torch.nn.functional as F

'''print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))
'''

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        random_tensor = random_tensor.type_as(x)
        print("Random keep ratio:", random_tensor.float().mean().item())
        return x * random_tensor / keep_prob


class ClassToken(nn.Module):
    def __init__(self, hidden_size):
        super(ClassToken, self).__init__()
        self.hidden_size = hidden_size
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls.expand(batch_size, -1, -1)
        return torch.cat((cls_tokens, x), dim=1)

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, x):
        positions = torch.arange(0, self.num_patches, device=x.device).unsqueeze(0)
        pos_embed = self.position_embedding(positions)
        return x + pos_embed

class Prompts(nn.Module):
    def __init__(self, projection_dims, prompt_count=1):
        super(Prompts, self).__init__()
        self.projection_dims = projection_dims
        self.prompt_count = prompt_count
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, projection_dims)) for _ in range(prompt_count)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        prompt_cat = torch.cat([p.expand(batch_size, -1, -1) for p in self.prompts], dim=1)
        return torch.cat((x, prompt_cat), dim=1)

class GatedLinearUnit(nn.Module):
    def __init__(self, dim, expansion_factor=2, dropout=0.0):
        super(GatedLinearUnit, self).__init__()
        hidden_dim = dim * expansion_factor
        self.fc1 = nn.Linear(dim, hidden_dim * 2)  # 输出为两个 hidden_dim 的拼接
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.act(gate)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SensorPatchesTimeDistributed(nn.Module):
    def __init__(self, projection_dim, filter_count, patch_count, frame_size=128, channels_count=6):
        super().__init__()
        self.projection_dim = projection_dim
        self.filter_count = filter_count
        self.patch_count = patch_count
        self.frame_size = frame_size
        self.channels_count = channels_count

        self.kernel_size = (projection_dim // 2 + filter_count) // filter_count
        assert ((projection_dim // 2 + filter_count) / filter_count) % self.kernel_size == 0, \
            "Kernel size condition not satisfied."

        self.reshape = lambda x: x.view(x.size(0), patch_count, frame_size // patch_count, channels_count)

        self.acc_projection = nn.Conv1d(3, filter_count, kernel_size=self.kernel_size, stride=1)
        self.gyro_projection = nn.Conv1d(3, filter_count, kernel_size=self.kernel_size, stride=1)

    def forward(self, x):
        # x shape: (B, T, C=6)
        x = self.reshape(x)  # → (B, P, F, C)
        B, P, F, C = x.shape

        acc = x[..., :3].reshape(B * P, 3, F)       # (B*P, 3, F)
        gyro = x[..., 3:].reshape(B * P, 3, F)       # (B*P, 3, F)

        acc_proj = self.acc_projection(acc)          # (B*P, filter_count, L)
        gyro_proj = self.gyro_projection(gyro)

        acc_proj = acc_proj.reshape(B, P, -1)
        gyro_proj = gyro_proj.reshape(B, P, -1)

        projections = torch.cat((acc_proj, gyro_proj), dim=2)
        return projections

class SensorWiseMHA(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0, start_index=None, stop_index=None, drop_path_rate=0.0):
        super(SensorWiseMHA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.start_index = start_index
        self.stop_index = stop_index
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, return_attention_scores=False):
        # x: (B, L, C)
        if self.start_index is not None and self.stop_index is not None:
            x = x[:, :, self.start_index:self.stop_index]

        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3*D)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, L, head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, L, L)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ v  # (B, num_heads, L, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(attn_output)
        out = self.drop_path(out)

        if return_attention_scores:
            return out, attn_weights
        else:
            return out
        
class LiteFormer(nn.Module):
    def __init__(self, start_index, stop_index, projection_size, kernel_size=16, attention_head=3, use_bias=False, drop_path_rate=0.0):
        super().__init__()
        self.start_index = start_index
        self.stop_index = stop_index
        self.kernel_size = kernel_size
        self.attention_head = attention_head
        self.use_bias = use_bias
        self.projection_size = projection_size
        self.drop_path = DropPath(drop_path_rate)

        # depthwise kernels: (head, 1, 1, kernel_size)
        self.depthwise_kernels = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, kernel_size)) for _ in range(attention_head)
        ])
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(attention_head))

    def forward(self, x):
        # x: (B, L, D)
        x = x[:, :, self.start_index:self.stop_index]  # slice dimension
        B, L, D = x.shape
        H = self.attention_head
        assert D == H, f"Expected D ({D}) == attention heads ({H})"

        # Reshape: (B, L, H) → (B*H, 1, L)
        x = x.permute(0, 2, 1).reshape(B * H, 1, L)

        # Normalize depthwise kernels with softmax across kernel_size
        soft_kernels = [F.softmax(k, dim=-1) for k in self.depthwise_kernels]

        outputs = []
        for i in range(H):
            k = soft_kernels[i]
            out = F.conv1d(x[i:i+1], k.unsqueeze(1), padding=self.kernel_size // 2)
            outputs.append(out)

        out = torch.cat(outputs, dim=0).reshape(B, H, L).permute(0, 2, 1)  # → (B, L, D)
        out = self.drop_path(out)
        return out
    
class MixAccGyro(nn.Module):
    def __init__(self, projection_quarter, projection_half, projection_dim):
        super(MixAccGyro, self).__init__()
        self.projection_quarter = projection_quarter
        self.projection_half = projection_half
        self.projection_dim = projection_dim

        mixed = []
        for i in range(projection_quarter):
            mixed.append(projection_quarter + i)
            mixed.append(projection_half + i)

        mixed_acc_gyro = torch.tensor(mixed, dtype=torch.long)
        head = torch.arange(0, projection_quarter, dtype=torch.long)
        tail = torch.arange(projection_half + projection_quarter, projection_dim, dtype=torch.long)

        self.register_buffer('new_order', torch.cat([head, mixed_acc_gyro, tail], dim=0))

    def forward(self, x):
        # x shape: (B, L, D)
        return x[:, :, self.new_order]

class MLP(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        layers = []
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MLP2(nn.Module):
    def __init__(self, in_dim, hidden1, hidden2, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden1, hidden2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DepthMLP(nn.Module):
    def __init__(self, in_dim, hidden1, hidden2, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.depthwise = nn.Conv1d(hidden1, hidden1, kernel_size=3, padding=1, groups=hidden1)
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (B, L, D) → transpose for conv: (B, D, L)
        x = self.fc1(x)
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x
    

class SensorPatches(nn.Module):
    def __init__(self, projection_dim, patch_size, time_step):
        super().__init__()
        self.patch_size = patch_size
        self.time_step = time_step
        self.projection_dim = projection_dim

        self.acc_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 2,
            kernel_size=patch_size,
            stride=time_step
        )
        self.gyro_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 2,
            kernel_size=patch_size,
            stride=time_step
        )

    def forward(self, x):
        # x: (B, T, 6)
        acc = x[:, :, :3].transpose(1, 2)   # (B, 3, T)
        gyro = x[:, :, 3:].transpose(1, 2)    # (B, 3, T)
 
        acc_proj = self.acc_projection(acc)   # (B, projection_dim/2, L)
        gyro_proj = self.gyro_projection(gyro)  # (B, projection_dim/2, L)
 
        out = torch.cat((acc_proj, gyro_proj), dim=1)  # (B, projection_dim, L)
        return out.transpose(1, 2)  # (B, L, projection_dim)

class ThreeSensorPatches(nn.Module):
    def __init__(self, projection_dim, patch_size, time_step):
        super().__init__()
        self.patch_size = patch_size
        self.time_step = time_step
        self.projection_dim = projection_dim

        self.acc_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 3,
            kernel_size=patch_size,
            stride=time_step
        )
        self.gyro_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 3,
            kernel_size=patch_size,
            stride=time_step
        )
        self.mag_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 3,
            kernel_size=patch_size,
            stride=time_step
        )

    def forward(self, x):
        # x: (B, T, 9) → acc, gyro, mag
        acc = x[:, :, :3].transpose(1, 2)
        gyro = x[:, :, 3:6].transpose(1, 2)
        mag = x[:, :, 6:9].transpose(1, 2)

        acc_proj = self.acc_projection(acc)
        gyro_proj = self.gyro_projection(gyro)
        mag_proj = self.mag_projection(mag)

        out = torch.cat((acc_proj, gyro_proj, mag_proj), dim=1)
        return out.transpose(1, 2)  # (B, L, projection_dim)
    
class FourSensorPatches(nn.Module):
    def __init__(self, projection_dim, patch_size, time_step):
        super().__init__()
        self.patch_size = patch_size
        self.time_step = time_step
        self.projection_dim = projection_dim

        self.acc_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 4,
            kernel_size=patch_size,
            stride=time_step
        )
        self.gyro_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 4,
            kernel_size=patch_size,
            stride=time_step
        )
        self.mag_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 4,
            kernel_size=patch_size,
            stride=time_step
        )
        self.alt_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 4,
            kernel_size=patch_size,
            stride=time_step
        )

    def forward(self, x):
        # x: (B, T, 12) → acc, gyro, mag, alt
        acc = x[:, :, :3].transpose(1, 2)
        gyro = x[:, :, 3:6].transpose(1, 2)
        mag = x[:, :, 6:9].transpose(1, 2)
        alt = x[:, :, 9:12].transpose(1, 2)

        acc_proj = self.acc_projection(acc)
        gyro_proj = self.gyro_projection(gyro)
        mag_proj = self.mag_projection(mag)
        alt_proj = self.alt_projection(alt)

        out = torch.cat((acc_proj, gyro_proj, mag_proj, alt_proj), dim=1)
        return out.transpose(1, 2)  # (B, L, projection_dim)

def extract_intermediate_model(model, layer_name):
    """
    构建一个新的模型，输出指定中间层的结果。
    layer_name: str，例如 'encoder.3'（假设模块是嵌套结构）
    """
    class IntermediateModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.target_layer = layer_name

        def forward(self, x):
            outputs = {}
            def hook(module, input, output):
                outputs[self.target_layer] = output

            # 注册 hook
            handle = dict(self.base.named_modules())[self.target_layer].register_forward_hook(hook)
            _ = self.base(x)
            handle.remove()
            return outputs[self.target_layer]

    return IntermediateModel(model)


class MazeruHAR(nn.Module):
    """
    PyTorch 版本的 HART 模型骨架，计划支持 Mamba 与 Transformer 的混合结构。
    后续将在此基础上逐步迁移原始 HART 模型结构并融合 Mamba 块。
    """
    def __init__(self, input_shape, activity_count, projection_dim=192, patch_size=16, time_step=16,
                 num_heads=3, filter_attention_head=4, conv_kernels=[3, 7, 15, 31, 31, 31],
                 mlp_head_units=[1024], dropout_rate=0.3, use_tokens=False):
        super(MazeruHAR, self).__init__()
        # 暂时保留结构参数，后续逐步实现 forward 逻辑
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
        self.num_heads = num_heads
        self.filter_attention_head = filter_attention_head
        self.conv_kernels = conv_kernels
        self.mlp_head_units = mlp_head_units
        self.dropout_rate = dropout_rate
        self.use_tokens = use_tokens

    def forward(self, x):
        # x: (B, T, C=6)
        B = x.size(0)
        projection_dim = self.projection_dim
        projection_half = projection_dim // 2
        projection_quarter = projection_dim // 4
        drop_path_rates = torch.linspace(0, self.dropout_rate * 10, steps=len(self.conv_kernels)) * 0.1
 
        # Patch Embedding
        x = SensorPatches(self.projection_dim, self.patch_size, self.time_step)(x)
        if self.use_tokens:
            x = ClassToken(self.projection_dim)(x)
        patch_count = x.size(1)
        x = PatchEncoder(patch_count, self.projection_dim)(x)
 
        # Transformer-like block layers
        for i, kernel_size in enumerate(self.conv_kernels):
            drop_path_rate = drop_path_rates[i].item()
 
            # LayerNorm
            x1 = F.layer_norm(x, (projection_dim,), eps=1e-6)
 
            # Branch 1: LiteFormer (middle quarter to three-quarter projection)
            branch1 = LiteFormer(
                start_index=projection_quarter,
                stop_index=projection_quarter + projection_half,
                projection_size=projection_half,
                attention_head=self.filter_attention_head,
                kernel_size=kernel_size,
                drop_path_rate=drop_path_rate,
            )(x1)
 
            # Branch 2: SensorWiseMHA for Acc and Gyro parts
            branch2_acc = SensorWiseMHA(
                dim=projection_quarter,
                num_heads=self.num_heads,
                start_index=0,
                stop_index=projection_quarter,
                dropout=self.dropout_rate,
                drop_path_rate=drop_path_rate,
            )(x1)
 
            branch2_gyro = SensorWiseMHA(
                dim=projection_quarter,
                num_heads=self.num_heads,
                start_index=projection_quarter + projection_half,
                stop_index=projection_dim,
                dropout=self.dropout_rate,
                drop_path_rate=drop_path_rate,
            )(x1)
 
            # Concatenate branches
            concat_attention = torch.cat([branch2_acc, branch1, branch2_gyro], dim=2)
 
            # Residual connection
            x2 = x + concat_attention
            x3 = F.layer_norm(x2, (projection_dim,), eps=1e-6)
            x3 = MLP2(projection_dim, projection_dim * 2, projection_dim, self.dropout_rate)(x3)
            x3 = DropPath(drop_path_rate)(x3)
            x = x2 + x3
 
        # Final representation
        x = F.layer_norm(x, (projection_dim,), eps=1e-6)
        if self.use_tokens:
            x = x[:, 0]  # Use class token
        else:
            x = x.mean(dim=1)  # Global average pooling
 
        x = MLP(self.mlp_head_units + [self.activity_count], self.dropout_rate)(x)
        return F.log_softmax(x, dim=-1)