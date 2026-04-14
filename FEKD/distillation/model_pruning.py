import torch
import torch.nn as nn
from torch.nn.utils import prune
from models.birefnet import BiRefNet  # 导入模型
from dataset import MyData  # 数据集
import numpy

# 定义 config 对象
class Config:
    pass

config = Config()

def collect_activation_statistics(model, dataloader, device):
    """
    收集模型中各层的激活统计信息
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        activations: 每层激活信息
    """
    model.eval()
    activations = {}

    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activation = output.detach().abs().mean(dim=(0, 2, 3)).cpu()
        else:
            activation = output[0].detach().abs().mean(dim=(0, 2, 3)).cpu()  # 处理可能的tuple输出
        if module not in activations:
            activations[module] = activation
        else:
            activations[module] += activation

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    with torch.no_grad():
        for inputs, _, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)

    for hook in hooks:
        hook.remove()

    return activations

def dynamic_channel_prune(model, activations, amount=0.5):
    """
    动态通道剪枝方法

    Args:
        model: 模型
        activations: 各层的激活统计信息
        amount: 剪枝比例

    Returns:
        剪枝后的模型
    """

    def prune_conv_layer(layer, activation, amount):
        """
        剪枝单个卷积层

        Args:
            layer: 卷积层
            activation: 该层的激活统计信息
            amount: 剪枝比例
        """
        num_channels = layer.weight.size(0)
        prune_channels = int(num_channels * amount)

        if prune_channels > 0:
            prune_indices = torch.argsort(activation)[:prune_channels]
            mask = torch.ones(num_channels)
            mask[prune_indices] = 0
            mask = mask.to(layer.weight.device)

            layer.weight.data.mul_(mask[:, None, None, None])
            if layer.bias is not None:
                layer.bias.data.mul_(mask)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module_name = str(module)  # 使用模块名称
            if module_name in activations:
                prune_conv_layer(module, activations[module_name], amount)

    return model

def prune_model(model, amount=0.5):
    """
    全局剪枝方法

    Args:
        model: 模型
        amount： 剪枝比例

    Return:
        返回剪枝后的模型
    """

    parameters_to_prune = []

    # 为所有 Conv2d 层和 Linear 层添加剪枝参数
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # 使用全局剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    # 将剪枝的参数应用到模型中
    for module, param in parameters_to_prune:
        prune.remove(module, 'weight')

    return model

# 查看为0的参数
def count_zero_weights(model):
    zero_params = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()
    return zero_params, total_params

def main():
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载预训练模型
    model = BiRefNet(bb_pretrained=True)
    # 加载权重文件
    state_dict = torch.load('../ckpt/C1/BiRefNet-COD-epoch_125.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # 加载数据集
    train_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=1024, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 收集激活统计信息
    activations = collect_activation_statistics(model, train_loader, device)

    # 打印原始模型参数量
    original_params = sum(p.numel() for p in model.parameters())
    print(f"原始模型参数量: {original_params}")

    # 动态通道剪枝
    #pruned_model = dynamic_channel_prune(model, activations, amount=0.5)

    # 全局剪枝
    pruned_model = prune_model(model, amount=0.3)

    # 打印剪枝后模型参数量
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    print(f"剪枝后模型参数量: {pruned_params}")

    # 统计剪枝后零值权重数量
    zero_params, total_params = count_zero_weights(pruned_model)
    print(f"剪枝后模型零值权重数量: {zero_params}")
    print(f"剪枝后模型零值权重比例: {zero_params / total_params:.2f}")

    # 保存剪枝后的模型
    torch.save(pruned_model.state_dict(), '../prune/pruned_125_0_3.pth')

if __name__ == "__main__":
    main()