import os
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm
import cv2
import torch

from dataset import MyData
from models.birefnet import BiRefNet
from utils import save_tensor_img, check_state_dict
from config import Config
from distillation.student import ResNet101Student
from distillation.studentResNet50 import ResNet50Student
# from distillation.student import MobileNetV2Student
from distillation.EffectNet_student import EfficientNetB2Student

config = Config()
#pred_root : ./e_preds
def parse_model_output(output):
    """兼容蒸馏模型输出，返回 (N, 1, H, W) 的预测 tensor"""
    if isinstance(output, (list, tuple)):
        # 找到第一个 tensor
        for o in output:
            if isinstance(o, torch.Tensor):
                pred = o
                break
        else:
            pred = output[0] if isinstance(output[0], torch.Tensor) else output[0][0]
    elif isinstance(output, torch.Tensor):
        pred = output
    else:
        raise TypeError(f"Unsupported model output type: {type(output)}")

    # shape 规范化为 (N, 1, H, W)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
    elif pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)

    return pred

def inference(model, data_loader_test, pred_root, method, testset, device=0):
    model_training = model.training
    if model_training:
        model.eval()
    for batch in tqdm(data_loader_test, total=len(data_loader_test)) if 1 or config.verbose_eval else data_loader_test:
        inputs = batch[0].to(device)
        # gts = batch[1].to(device)
        label_paths = batch[-1]
        with torch.no_grad():

            # scaled_preds = model(inputs)[-1].sigmoid()
######lsnmini教师频域-------注释上面一行--------------------------------
            output = model(inputs)
            pred = parse_model_output(output)
            scaled_preds = pred.sigmoid()

        # print(f"Predicted shape: {scaled_preds.shape}")

        os.makedirs(os.path.join(pred_root, method, testset), exist_ok=True)

        for idx_sample in range(scaled_preds.shape[0]):
            original_shape = cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2]  # 获取原始图像的形状
            # print(f"Input shape: {scaled_preds[idx_sample].unsqueeze(0).shape}, Output shape: {original_shape}")  #打印输入和输出的形状

            res = torch.nn.functional.interpolate(
                scaled_preds[idx_sample:idx_sample + 1],  # shape (1, 1, H, W)
                size=original_shape,  # (H, W)
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (1, H, W) -> (H, W)

            # res = torch.nn.functional.interpolate(
            #     scaled_preds[idx_sample].unsqueeze(0),
            #     size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
            #     mode='bilinear',
            #     align_corners=True
            # )

            save_tensor_img(res, os.path.join(os.path.join(pred_root, method, testset), label_paths[idx_sample].replace('\\', '/').split('/')[-1]))   # test set dir + file name

    if model_training:
        model.train()
    return None


def main(args):
    # Init model

    # 设置设备
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config.device
    if args.ckpt_folder:
        print('Testing with models in {}'.format(args.ckpt_folder))
    else:
        print('Testing with model {}'.format(args.ckpt))

    # if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=False)

    model = ResNet101Student(num_classes=8)
    #model = ResNet50Student(num_classes=8)

    # weights_lst = sorted(
    #     glob(os.path.join(args.ckpt_folder, '*.pth')) if args.ckpt_folder else [args.ckpt],
    #     key=lambda x: int(x.split('epoch_')[-1].split('.pth')[0]),
    #     reverse=True
    # )

    #lsn优先使用推理指定权重文件----------------------------------
    # 优先使用 --ckpt 参数（单模型）
    # if args.ckpt is not None:
    #     weights_lst = [args.ckpt]
    # else:
    #     weights_lst = sorted(
    #         glob(os.path.join(args.ckpt_folder, '*.pth')),
    #         key=lambda x: int(x.split('epoch_')[-1].split('.pth')[0]),
    #         reverse=True
    #     )
    # 优先使用 --ckpt 参数（单模型）
    if args.ckpt is not None:
        weights_lst = [args.ckpt]
    else:
        # 自动读取 ckpt_folder 下所有 student_epoch_*.pth
        weights_lst = sorted(
            glob(os.path.join(args.ckpt_folder, 'student_epoch_*.pth')),
            key=lambda x: int(os.path.basename(x).split('_')[-1].split('.pth')[0])
        )

    #-----------------------------------------------------------------
    #####
    print(f"Checkpoint folder: {args.ckpt_folder}")

    for testset in args.testsets.split('+'):
        print('>>>> Testset: {}...'.format(testset))
        data_loader_test = torch.utils.data.DataLoader(
            dataset=MyData(testset,
                           image_size=config.size,
                           is_train=False,
                           # custom_dataset_path=args.custom_dataset_path#lsn
                           ),
            batch_size=config.batch_size_valid, shuffle=False, num_workers=config.num_workers, pin_memory=True
        )
        for weights in weights_lst:
            if int(weights.strip('.pth').split('epoch_')[-1]) % 1 != 0:
                continue
            print('\tInferencing {}...'.format(weights))
            # model.load_state_dict(torch.load(weights, map_location='cpu'))
            state_dict = torch.load(weights, map_location='cpu')
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            model = model.to(device)
            inference(
                model, data_loader_test=data_loader_test, pred_root=args.pred_root,
                method='--'.join([w.rstrip('.pth') for w in weights.split(os.sep)[-2:]]),
                testset=testset, device=config.device
            )


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', type=str, help='model folder')
    parser.add_argument('--ckpt_folder', default=sorted(glob(os.path.join('ckpt', '*')))[-1], type=str, help='model folder')
    parser.add_argument('--pred_root', default='e_preds', type=str, help='Output folder')
    # parser.add_argument('--testsets',
    #                     default={
    #                         'DIS5K': 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4',
    #                         'COD': 'TE-COD10K+NC4K+TE-CAMO+CHAMELEON',
    #                         'HRSOD': 'DAVIS-S+TE-HRSOD+TE-UHRSD+TE-DUTS+DUT-OMRON',
    #                         'DIS5K+HRSOD+HRS10K': 'DIS-VD',
    #                         'P3M-10k': 'TE-P3M-500-P+TE-P3M-500-NP',
    #                         'DIS5K-': 'DIS-VD',
    #                         'COD-': 'TE-COD10K',
    #                         'SOD-': 'DAVIS-S+TE-HRSOD+TE-UHRSD',
    #                     }[config.task + ''],
    #                     type=str,
    #                     help="Test all sets: , 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'")


    parser.add_argument('--testsets',
                        default={
                            # 'DIS5K': 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4',
                            'COD': 'COD10K-v3+CAMO',
                            # 'HRSOD': 'DAVIS-S+TE-HRSOD+TE-UHRSD+TE-DUTS+DUT-OMRON',
                            # 'DIS5K+HRSOD+HRS10K': 'DIS-VD',
                            # 'P3M-10k': 'TE-P3M-500-P+TE-P3M-500-NP',
                            # 'DIS5K-': 'DIS-VD',
                            'COD-': 'COD10K-v3',
                            # 'SOD-': 'DAVIS-S+TE-HRSOD+TE-UHRSD',
                        }[config.task + ''],
                        type=str,
                        help="Test all sets: , 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'")
    parser.add_argument('--custom_dataset_path', type=str, help='Custom dataset path for inference')

#    ###lsn只检测CAMO数据集-------------------------------------------------
#     parser.add_argument('--testsets',
#                         default='CAMO',
#                         type=str,
#                         help="Test set: CAMO")
# #####-------------------------------------------------------------------------------
    args = parser.parse_args()

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')
    main(args)
