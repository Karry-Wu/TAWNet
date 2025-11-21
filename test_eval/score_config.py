# coding: utf-8
import os
import sys
# sys.path.append('../models')
sys.path.append('..')
from models.TAWNet import TAWNet

dutrgbd_root_test = '../dataset/RGB-D_test/DUT-RGBD/'   #
njud_root_test = '../dataset/RGB-D_test/NJUD/'    #
nlpr_root_test = '../dataset/RGB-D_test/NLPR/'    #
stere_root_test = '../dataset/RGB-D_test/STERE/'  #
sip_root_test = '../dataset/RGB-D_test/SIP/'      #
rgbd135_root_test = '../dataset/RGB-D_test/RGBD135/'  #DES
ssd_root_test = '../dataset/RGB-D_test/SSD/'
lfsd_root_test = '../dataset/RGB-D_test/LFSD/'

vt821_root_test = '../dataset/RGBT_dataset/test/VT821/'
vt1000_root_test = '../dataset/RGBT_dataset/test/VT1000/'
vt5000_root_test = '../dataset/RGBT_dataset/test/VT5000/'

dutrgbd = os.path.join(dutrgbd_root_test)
njud = os.path.join(njud_root_test)
nlpr = os.path.join(nlpr_root_test)
stere = os.path.join(stere_root_test)
sip = os.path.join(sip_root_test)
rgbd135 = os.path.join(rgbd135_root_test)
ssd = os.path.join(ssd_root_test)
lfsd = os.path.join(lfsd_root_test)

vt821 = os.path.join(vt821_root_test)
vt1000 = os.path.join(vt1000_root_test)
vt5000 = os.path.join(vt5000_root_test)

results_path = '../results/'
results_save_path = os.path.join('..', 'results/')



def getNet():
    return TAWNet(None).cuda()


def infer(net, rgb_trans, m_trans):
    side_out4, side_out3, side_out2, side_out1 = net(rgb_trans, m_trans)
    return side_out1
