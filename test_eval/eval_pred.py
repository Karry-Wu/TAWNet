import datetime
import os
import cv2
import numpy as np
import sys
import argparse
sys.path.append('..')
from load_test_data import test_dataset
from py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, MAE, Fmeasure
from score_config import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--modal', default='rgbd', type=str, help='rgbd or rgbt')
parser.add_argument('--p', default='TAWNet', type=str, help='model name, salient map path')
opt = parser.parse_args()
model_name = 'TAWNet'
if opt.modal == 'rgbd':
    test_datasets = {'DUT-RGBD': dutrgbd, 'NJUD': njud, 'NLPR': nlpr, 'STERE': stere, 'SIP': sip,
                     'RGBD135': rgbd135}
elif opt.modal == 'rgbt':
    test_datasets = {'VT-821': vt821, 'VT-1000': vt1000, 'VT-5000': vt5000}
else:
    print('please input rgbd or rgbt')
    exit()

results_save_path = 'ablation/pred/'+model_name
RGBD_SOD_Models = {opt.p: os.path.join(results_save_path, opt.p)}

if not os.path.exists(results_save_path):
    os.makedirs(results_save_path)

results_save_path = os.path.join(results_save_path, opt.p+'_metrics.txt')

open(results_save_path, 'a').write('\n' + str(datetime.datetime.now()) + '\n'+opt.p+'\n')
mae_list, max_f_list, sm_list, em_list, wfm_list = [], [], [], [], []

nums_ever_dataset = []

for method_name, method_map_root in RGBD_SOD_Models.items():
    print('test method:', method_name, method_map_root)
    for name, root in test_datasets.items():
        print('eval_'+name)
        sal_root = method_map_root + '/' + name
        print(sal_root)
        gt_root = root + 'GT'
        print(gt_root)
        if os.path.exists(sal_root):
            print('\033[32m file exist! \033[0m')
            test_loader = test_dataset(sal_root, gt_root)
            size = test_loader.size
            nums_ever_dataset.append(size)
            mae = MAE()
            wfm = WeightedFmeasure()
            sm = Smeasure()
            em = Emeasure()
            fm = Fmeasure()
            images = os.listdir(sal_root)
            for image in tqdm(images):

                gt = cv2.imread(os.path.join(gt_root, image), 0)
                predict = cv2.imread(os.path.join(sal_root, image), 0)

                mae.step(predict, gt)
                wfm.step(predict, gt)
                sm.step(predict, gt)
                em.step(predict, gt)
                fm.step(predict, gt)
            MAE_ = mae.get_results()['mae']
            mae_list.append(MAE_)
            maxf_= fm.get_results()['fm']['curve'].max()
            max_f_list.append(maxf_)
            sm_ = sm.get_results()['sm']
            sm_list.append(sm_)
            em_ = em.get_results()['em']['curve'].max()
            em_list.append(em_)
            wfm_ = wfm.get_results()['wfm']
            wfm_list.append(wfm_)
            log = 'method:{} dataset:{:>8}  Sm:{:.4f}  maxF:{:.4f}  Em:{:.4f}  MAE:{:.4f}  wfm:{:.4f}'.format(
                method_name, name, sm_, maxf_, em_, MAE_, wfm_)
            print(log+'\n')
            table_content = name + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' \\\\'
            table_content = table_content.format(MAE_, maxf_, wfm_, sm_, em_)
            logtxt = 'dataset:{:>8}  Sm:{:.4f}  maxF:{:.4f}  Em:{:.4f}  MAE:{:.4f}  wfm:{:.4f}'.format(
                name, sm_, maxf_, em_, MAE_, wfm_)
            open(results_save_path, 'a').write(logtxt + '\n')
        else:
            print('\033[31m file is not exist! \033[0m')

    mae, max_f, wfm, sm, em = np.array(mae_list), np.array(max_f_list), np.array(wfm_list), np.array(
        sm_list), np.array(em_list)
    avg_mae, avg_max_f, avg_wfm, avg_sm, avg_em = np.mean(mae), np.mean(max_f), np.mean(wfm), np.mean(
        sm), np.mean(em)

    nums_ever_dataset = np.array(nums_ever_dataset)
    nums_sample = nums_ever_dataset.sum()
    w_avg_mae, w_avg_max_f, w_avg_wfm, w_avg_sm, w_avg_em = nums_ever_dataset * mae, nums_ever_dataset * max_f, nums_ever_dataset * wfm, nums_ever_dataset * sm, nums_ever_dataset * em

    w_avg_mae, w_avg_max_f, w_avg_wfm, w_avg_sm, w_avg_em = np.sum(w_avg_mae/nums_sample), np.sum(w_avg_max_f/nums_sample), np.sum(w_avg_wfm/nums_sample), np.sum(w_avg_sm/nums_sample), np.sum(w_avg_em/nums_sample)

    avg_log = 'method_name: {} on all dataset avg_MAE: {:.4f} avg_maxF: {:.4f} avg_wfm: {:.4f} avg_Sm: {:.4f} avg_Em: {:.4f}'.format(
        method_name, avg_mae, avg_max_f, avg_wfm, avg_sm, avg_em)

    w_avg_log = 'method_name: {} on all dataset w_avg_MAE: {:.4f} avg_maxF: {:.4f} avg_wfm: {:.4f} avg_Sm: {:.4f} avg_Em: {:.4f}'.format(
        method_name, w_avg_mae, w_avg_max_f, w_avg_wfm, w_avg_sm, w_avg_em)
    print(avg_log)
    print(w_avg_log)

    table_avg = 'average' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' \\\\'
    table_w_avg = 'w_average' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' \\\\'
    table_avg = table_avg.format(avg_mae, avg_max_f, avg_wfm, avg_sm, avg_em)
    table_w_avg = table_w_avg.format(w_avg_mae, w_avg_max_f, w_avg_wfm, w_avg_sm, w_avg_em)

