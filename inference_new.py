import os
import argparse
from glob import glob

import torch
from tqdm import tqdm
import cv2

from dataset import MyData
from models.birefnet import BiRefNet
from utils import save_tensor_img, check_state_dict
from config import Config
from distillation.student import ResNet101Student
from distillation.studentResNet50 import ResNet50Student
from distillation.EffectNet_student import EfficientNetB2Student

config = Config()


def parse_model_output(output):
    """
    解析模型输出，保证返回 tensor 预测值 (N, 1, H, W)
    """
    if isinstance(output, (list, tuple)):
        # 常见输出格式 [pred, feats...]
        for o in output:
            if isinstance(o, torch.Tensor):
                pred = o
                break
        else:
            # fallback
            pred = output[0] if isinstance(output[0], torch.Tensor) else output[0][0]
    elif isinstance(output, torch.Tensor):
        pred = output
    else:
        raise TypeError(f"Unsupported model output type: {type(output)}")

    # 确保 shape 为 (N, C, H, W)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    elif pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)

    return pred


def inference(model, data_loader, pred_root, method, testset, device='cuda'):
    training_mode = model.training
    if training_mode:
        model.eval()

    os.makedirs(os.path.join(pred_root, method, testset), exist_ok=True)

    for batch in tqdm(data_loader, total=len(data_loader)):
        images = batch[0].to(device)
        label_paths = batch[-1]

        with torch.no_grad():
            output = model(images)
            pred = parse_model_output(output)
            pred = pred.to(device)
            # 对输出进行 sigmoid
            pred = pred.sigmoid()

        # 遍历 batch 中每个样本
        for i in range(pred.shape[0]):
            orig_img_path = label_paths[i]
            orig_img = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = orig_img.shape[:2]

            # 插值到原始图像尺寸
            res = torch.nn.functional.interpolate(
                pred[i].unsqueeze(0),  # (1, 1, H, W)
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (1, H, W) -> (H, W)

            save_path = os.path.join(pred_root, method, testset,
                                     os.path.basename(orig_img_path))
            save_tensor_img(res, save_path)

    if training_mode:
        model.train()


def main(args):
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化模型
    if args.model_type == 'BiRefNet':
        model = BiRefNet(bb_pretrained=False)
    elif args.model_type == 'ResNet101Student':
        model = ResNet101Student(num_classes=8)
    elif args.model_type == 'ResNet50Student':
        model = ResNet50Student(num_classes=8)
    elif args.model_type == 'EfficientNetB2Student':
        model = EfficientNetB2Student(num_classes=8)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # 获取权重列表
    if args.ckpt is not None:
        weights_lst = [args.ckpt]
    else:
        weights_lst = sorted(
            glob(os.path.join(args.ckpt_folder, '*.pth')),
            key=lambda x: int(x.split('epoch_')[-1].split('.pth')[0]),
            reverse=True
        )

    for testset in args.testsets.split('+'):
        print(f">>>> Testset: {testset}...")

        data_loader = torch.utils.data.DataLoader(
            dataset=MyData(testset,
                           image_size=config.size,
                           is_train=False),
            batch_size=config.batch_size_valid,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        for weights in weights_lst:
            print(f'\tInferencing {weights}...')
            state_dict = torch.load(weights, map_location='cpu')
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            model = model.to(device)

            method_name = '--'.join([w.rstrip('.pth') for w in weights.split(os.sep)[-2:]])
            inference(model, data_loader, pred_root=args.pred_root,
                      method=method_name, testset=testset, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for COD student models')
    parser.add_argument('--ckpt', type=str, help='Single checkpoint file')
    parser.add_argument('--ckpt_folder', type=str, default='ckpt', help='Checkpoint folder')
    parser.add_argument('--pred_root', type=str, default='e_preds', help='Output folder')
    parser.add_argument('--testsets', type=str, default='COD10K-v3+CAMO', help='Test datasets')
    parser.add_argument('--model_type', type=str, default='ResNet101Student',
                        choices=['BiRefNet', 'ResNet101Student', 'ResNet50Student', 'EfficientNetB2Student'])
    args = parser.parse_args()

    main(args)
