import os
import cv2#lsn
import argparse
from glob import glob

import numpy as np
import prettytable as pt

from evaluation.evaluate import evaluator
from config import Config


config = Config()


def do_eval(args):
    # evaluation for whole dataset
    # dataset first in evaluation
    for _data_name in args.data_lst.split('+'):
        # 获取预测结果路径
        pred_data_dir =  sorted(glob(os.path.join(args.pred_root, args.model_lst[0], _data_name)))
        if not pred_data_dir:
            print('Skip dataset {}.'.format(_data_name))
            continue
        gt_src = os.path.join(args.gt_root, _data_name)

        # gt_paths = sorted(glob(os.path.join(gt_src, 'gt', '*')))
        gt_paths = sorted(glob(os.path.join(gt_src, 'Train/GT_Instance', '*')))
        print('#' * 20, _data_name, '#' * 20)
        filename = os.path.join(args.save_dir, '{}_eval.txt'.format(_data_name))
        tb = pt.PrettyTable()
        tb.vertical_char = '&'
        if config.task == 'DIS5K':
            tb.field_names = ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm"]
        elif config.task == 'COD':
            tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "meanFm", "meanEm", "maxEm", 'MAE', "maxFm", "adpEm", "adpFm", "HCE"]
        elif config.task == 'HRSOD':
            tb.field_names = ["Dataset", "Method", "Smeasure", "maxFm", "meanEm", 'MAE', "maxEm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE"]
        elif config.task == 'DIS5K+HRSOD+HRS10K':
            tb.field_names = ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm"]
        elif config.task == 'P3M-10k':
            tb.field_names = ["Dataset", "Method", "Smeasure", "maxFm", "meanEm", 'MAE', "maxEm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE"]
        else:
            tb.field_names = ["Dataset", "Method", "Smeasure", 'MAE', "maxEm", "meanEm", "maxFm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE"]
        for _model_name in args.model_lst[:]:
            print('\t', 'Evaluating model: {}...'.format(_model_name))

            # pred_paths = [p.replace(args.gt_root, os.path.join(args.pred_root, _model_name)).replace('/gt/', '/') for p in gt_paths]#测试集评估
            pred_paths = [p.replace(args.gt_root, os.path.join(args.pred_root, _model_name)).replace('/Train/GT_Instance/', '/') for p in gt_paths]#训练集评估（检查训练效果或调试）
            print(pred_paths[:1], gt_paths[:1])

            em, sm, fm, mae, wfm, hce = evaluator(
                gt_paths=gt_paths,
                pred_paths=pred_paths,
                metrics=args.metrics.split('+'),
                verbose=config.verbose_eval
            )
#lsn注释上面10行改为下面-----------------------------------------------------
#             # ===== LSN 修正版：按文件名匹配 GT 与预测图像 =====
#             pred_dir = os.path.join(args.pred_root, _model_name, _data_name)
#             gt_dir = os.path.join(gt_src, 'Train/GT_Instance')
#
#
#             pred_files = {os.path.basename(p): p for p in glob(os.path.join(pred_dir, '*.png'))}
#             gt_files = {os.path.basename(p): p for p in glob(os.path.join(gt_dir, '*.png'))}
#
#             common_names = sorted(set(pred_files.keys()) & set(gt_files.keys()))
#
#             if len(common_names) == 0:
#                 print(f"❌ 错误：{_data_name}/{_model_name} 没有匹配的文件！")
#                 continue
#
#             valid_pairs = []
#             skipped_count = 0
#             black_gt_count = 0
#
#             for name in common_names:
#                 pred_path = pred_files[name]
#                 gt_path = gt_files[name]
#                 pred_ary = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
#                 gt_ary = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
#
#                 if pred_ary is None:
#                     print(f"⚠️ 跳过：预测图像无法读取 {name}")
#                     skipped_count += 1
#                     continue
#                 if gt_ary is None:
#                     print(f"⚠️ 跳过：GT无法读取 {name}")
#                     skipped_count += 1
#                     continue
#
#                 if np.max(gt_ary) == 0:
#                     black_gt_count += 1
#                     continue
#
#                 valid_pairs.append((pred_path, gt_path))
#
#             valid_pred_paths = [v[0] for v in valid_pairs]
#             valid_gt_paths = [v[1] for v in valid_pairs]
#
#             print(
#                 f"总图像: {len(common_names)} | 有效: {len(valid_pairs)} | 跳过损坏: {skipped_count} | 跳过全黑GT: {black_gt_count}")
#
#             if len(valid_pred_paths) == 0:
#                 print(f"❌ 错误：{_model_name} 没有有效图像可评估")
#                 continue
#
#             # ===== 开始评估 =====
#             em, sm, fm, mae, wfm, hce = evaluator(
#                 gt_paths=valid_gt_paths,
#                 pred_paths=valid_pred_paths,
#                 metrics=args.metrics.split('+'),
#                 verbose=config.verbose_eval
#             )
# ####----------------------------------------------------------------------------------




            if config.task == 'DIS5K':
                scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                ]
            elif config.task == 'COD':
                scores = [
                    sm.round(3), wfm.round(3), fm['curve'].mean().round(3), em['curve'].mean().round(3), em['curve'].max().round(3), mae.round(3),
                    fm['curve'].max().round(3), em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                ]
            elif config.task == 'HRSOD':
                scores = [
                    sm.round(3), fm['curve'].max().round(3), em['curve'].mean().round(3), mae.round(3),
                    em['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3), em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                ]
            elif config.task == 'DIS5K+HRSOD+HRS10K':
                scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                ]
            elif config.task == 'P3M-10k':
                scores = [
                    sm.round(3), fm['curve'].max().round(3), em['curve'].mean().round(3), mae.round(3),
                    em['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3), em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                ]
            else:
                scores = [
                    sm.round(3), mae.round(3), em['curve'].max().round(3), em['curve'].mean().round(3),
                    fm['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3),
                    em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                ]
            
            for idx_score, score in enumerate(scores):
                scores[idx_score] = '.' + format(score, '.3f').split('.')[-1] if score <= 1  else format(score, '<4')
            records = [_data_name, _model_name] + scores
            tb.add_row(records)
            # Write results after every check.
            with open(filename, 'w+') as file_to_write:
                file_to_write.write(str(tb)+'\n')
        print(tb)


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str, help='ground-truth root',
        default=os.path.join(config.data_root_dir, config.task))  # /home/user/lc/datasets/COD   --gt_root
    parser.add_argument(
        '--pred_root', type=str, help='prediction root',
        default='./e_preds')
    # parser.add_argument(
    #     '--data_lst', type=str, help='test dataset',
    #     default={
    #         'DIS5K': '+'.join(['DIS-VD', 'DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4'][:]),
    #         'COD_version1': '+'.join(['TE-COD10K', 'NC4K', 'TE-CAMO', 'CHAMELEON'][:]),
    #         'HRSOD': '+'.join(['DAVIS-S', 'TE-HRSOD', 'TE-UHRSD', 'TE-DUTS', 'DUT-OMRON'][:]),
    #         'DIS5K+HRSOD+HRS10K': '+'.join(['DIS-VD'][:]),
    #         'P3M-10k': '+'.join(['TE-P3M-500-P', 'TE-P3M-500-NP'][:]),
    #     }[config.task])
    parser.add_argument(
        '--data_lst', type=str, help='test dataset',
        default={
            'DIS5K': '+'.join(['DIS-VD', 'DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4'][:]),
            # 'COD': '+'.join(['COD10K-v3', 'CAMO'][:]),
            'COD': '+'.join(['COD10K-v3', 'CAMO'][:]),
            'HRSOD': '+'.join(['DAVIS-S', 'TE-HRSOD', 'TE-UHRSD', 'TE-DUTS', 'DUT-OMRON'][:]),
            'DIS5K+HRSOD+HRS10K': '+'.join(['DIS-VD'][:]),
            'P3M-10k': '+'.join(['TE-P3M-500-P', 'TE-P3M-500-NP'][:]),
        }[config.task])
    parser.add_argument(
        '--save_dir', type=str, help='candidate competitors',
        default='e_results')
    parser.add_argument(
        '--check_integrity', type=bool, help='whether to check the file integrity',
        default=False)  # 默认值为False，但是我强制检查完整性
    parser.add_argument(
        '--metrics', type=str, help='candidate competitors',
        default='+'.join(['S', 'MAE', 'E', 'F', 'WF', 'HCE'][:100 if 'DIS5K' in config.task else -1]))
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    try:
        args.model_lst = [m for m in sorted(os.listdir(args.pred_root), key=lambda x: int(x.split('epoch_')[-1]), reverse=True) if int(m.split('epoch_')[-1]) % 1 == 0]
    except:
        args.model_lst = [m for m in sorted(os.listdir(args.pred_root))]
        args.model_lst = [m for m in sorted(os.listdir(args.pred_root))]
    print(args.model_lst)  # ['epoch_10.pth', 'epoch_15.pth', 'epoch_20.pth', 'epoch_5.pth']权重文件
    # check the integrity of each candidates
    if args.check_integrity:
        for _data_name in args.data_lst.split('+'):
            for _model_name in args.model_lst:
                gt_pth = os.path.join(args.gt_root, _data_name)
                pred_pth = os.path.join(args.pred_root, _model_name, _data_name)
                if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                    print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                    print('The {} Dataset of {} Model is not matching to the ground-truth'.format(_data_name, _model_name))
    else:
        print('>>> skip check the integrity of each candidates')


    # start engine
    do_eval(args)
