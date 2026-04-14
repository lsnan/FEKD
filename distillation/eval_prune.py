import torch
from dataset import MyData
from torchvision import datasets
from student import MobileNetV2Student

def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()  # 损失函数
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
        return total_loss / len(val_loader)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    val_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2)

    # 加载微调后的模型
    model = SimpleUNet()
    model.load_state_dict(torch.load('../prune/finetuned_model_kd_30_0.001_0_5_T2.pth'))
    model.to(device)

    # 验证性能
    val_loss = validate_model(model, val_loader, device)
    print(f'Validation Loss:{val_loss}')

if __name__ == "__main__":
    main()