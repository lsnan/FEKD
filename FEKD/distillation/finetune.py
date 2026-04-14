import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp  # 使用混合精度训练
from dataset import MyData  # 数据集
from torchvision import datasets, transforms
from models.birefnet import BiRefNet  # 导入模型
#from simple_unet import SimpleUNet  # 导入自定义的学生模型

torch.cuda.empty_cache()  # 清理缓存

# 定义 config 对象
class Config:
    pass

config = Config()

def load_pruned_model(filepath):
    """
    加载剪枝后的模型
    """
    model = BiRefNet(bb_pretrained=False)
    model.load_state_dict(torch.load(filepath))
    return model

def extract_tensor_from_nested(outputs):
    """
    从嵌套结构中提取张量部分

    Args:
        outputs: 嵌套结构，包含张量 结构[tensor, None]

    Returns:
        张量部分
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    elif isinstance(outputs, list) or isinstance(outputs, tuple):
        for item in outputs:
            result = extract_tensor_from_nested(item)
            if result is not None:
                return result
    return None

def load_teacher_model(filepath, model):
    """
    加载教师模型的状态字典，忽略不存在的键
    """
    state_dict = torch.load(filepath)
    model_dict = model.state_dict()
    # 过滤掉不匹配的键
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

def knowledge_distillation_loss(outputs, teacher_outputs, labels, alpha=0.5, T=2.0):
    """
    知识蒸馏损失函数

    Args:
        outputs: 学生模型输出
        teacher_outputs: 教师模型输出
        labels: 真实标签
        alpha: 损失函数中蒸馏损失的权重
        T: 温度参数

    Returns:
        知识蒸馏损失
    """
    # 知识蒸馏损失
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(outputs / T, dim=1),
        nn.functional.softmax(teacher_outputs / T, dim=1)) * (T * T)

    # 交叉熵损失
    classification_loss = nn.BCEWithLogitsLoss()(outputs, labels)
    return alpha * distillation_loss + (1 - alpha) * classification_loss

def finetune_model_with_kd(student_model, teacher_model, train_loader, epochs=10, lr=0.001, alpha=0.5, T=2.0):
    """
    使用知识蒸馏和混合精度训练微调模型

    Args:
        student_model: 学生模型
        teacher_model: 教师模型
        train_loader: 训练数据的加载器
        epoch: 微调轮数
        lr: 学习率
        alpha: 损失函数中蒸馏损失的权重
        T: 温度参数
    """
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    scaler = amp.GradScaler()  # 使用混合精度

    student_model.train()
    teacher_model.eval()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels, _ in train_loader:  # 由于数据集返回（image, label, class_label），所以这里也要有三个
            # 将模型和数据移动到 GPU 上
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            optimizer.zero_grad()  # 每次将梯度归0

            with amp.autocast():
                student_outputs = student_model(inputs)
                teacher_outputs = teacher_model(inputs)

                student_outputs = extract_tensor_from_nested(student_outputs)  # 提取张量部分
                if student_outputs is None:
                    raise ValueError("无法从输出中提取张量部分")

                teacher_outputs = extract_tensor_from_nested(teacher_outputs)
                if teacher_outputs is None:
                    raise ValueError("无法从教师模型输出中提取张量部分")

                # 确保 student_outputs 是四维的 [N, C, H, W]
                if student_outputs.dim() == 3:
                    student_outputs = student_outputs.unsqueeze(1)
                elif student_outputs.dim() != 4:
                    raise ValueError(f"student_outputs 维度错误: {student_outputs.dim()}")

                # 确保 teacher_outputs 是四维的 [N, C, H, W]
                if teacher_outputs.dim() == 3:
                    teacher_outputs = teacher_outputs.unsqueeze(1)
                elif teacher_outputs.dim() != 4:
                    raise ValueError(f"teacher_outputs 维度错误: {teacher_outputs.dim()}")

                # 调整 student_outputs 的形状
                student_outputs = nn.functional.interpolate(student_outputs, size=(labels.size(2), labels.size(3)),
                                                            mode='bilinear', align_corners=False)

                # 确保输出和标签的通道数一致
                if student_outputs.size(1) != labels.size(1):
                    student_outputs = student_outputs[:, :labels.size(1)]

                # 调整 teacher_outputs 的形状
                teacher_outputs = nn.functional.interpolate(teacher_outputs, size=(labels.size(2), labels.size(3)),
                                                            mode='bilinear', align_corners=False)

                # 确保输出和标签的通道数一致
                if teacher_outputs.size(1) != labels.size(1):
                    teacher_outputs = teacher_outputs[:, :labels.size(1)]

                labels = labels.float()  # 将labels转换为float类型

                loss = knowledge_distillation_loss(student_outputs, teacher_outputs, labels, alpha, T)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            # 清理 CUDA 缓存
            torch.cuda.empty_cache()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

    return student_model

def main():
    # 设置设备
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据集
    train_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=True)  # 对应config.py中
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)

    # 学生模型
    student_model = SimpleUNet()

    # 加载预训练的教师模型  使用剪枝后的
    teacher_model = BiRefNet(bb_pretrained=False)
    load_teacher_model('../ckpt/C1/BiRefNet-COD-epoch_125.pth', teacher_model)
    teacher_model.to(config.device)

    # 将学生模型移动到 GPU
    student_model.to(config.device)

    # 微调模型
    finetuned_model = finetune_model_with_kd(student_model, teacher_model, train_loader, epochs=30, lr=0.001)

    # 保存微调模型
    torch.save(finetuned_model.state_dict(), 'prune/finetune_150.pth')

if __name__ == "__main__":
    main()
