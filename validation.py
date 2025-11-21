import os
import numpy as np
import tensorboardX
import torch
from PIL.Image import Image
from torchvision import transforms
from tqdm import tqdm
from data_loader.rgbd_validation_loader import valid_dataset
from PIL import Image
from test_eval.py_sod_metrics import MAE
from torchvision.utils import make_grid
from torch.nn import functional as F
from config import validation_root

def validation(model, input_size, sw, validation_step):

    to_test = {'validation': validation_root}

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    model.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            test_loader = valid_dataset(root, input_size)
            mae = MAE()
            for i in tqdm(range(test_loader.size)):
                image, depth, gt, HH, WW, image_name = test_loader.load_data()
                image = image.cuda()
                depth = depth.cuda()
                # depth = torch.cat((depth, depth, depth), 1)
                side_out4, side_out3, side_out2, side_out1 = model(image, depth)

                sal_map = side_out1
                sal_map = sal_map.sigmoid()
                sal_map = sal_map.data.cpu()
                validation_step += 1

                if validation_step % 100 == 0:
                    grid_image = make_grid([image[0], depth[0]], 2, normalize=True)
                    sw.add_image('validation_RGB_Depth', grid_image, validation_step)
                    sal_map_show = F.interpolate(sal_map, (WW, HH), mode='bilinear', align_corners=False)
                    sal_map_show = sal_map_show.squeeze().numpy()
                    sal_map_show = (sal_map_show - sal_map_show.min()) / (
                                sal_map_show.max() - sal_map_show.min() + 1e-8)
                    sal_map_show = torch.tensor(sal_map_show).unsqueeze(dim=0)
                    gt_show = to_tensor(gt)
                    grid_image = make_grid([gt_show, sal_map_show], 2, pad_value=1, normalize=True)
                    sw.add_image('validation_gt_res', grid_image, validation_step)

                sal_map = to_pil(sal_map.squeeze(dim=0))
                sal_map = sal_map.resize((HH, WW), Image.BILINEAR)
                if sal_map.size != gt.size:
                    x, y = gt.size
                    sal_map = sal_map.resize((x, y))
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                gt[gt > 0.5] = 1
                gt[gt != 1] = 0
                res = sal_map
                res = np.array(res)
                if res.max() == res.min():
                    res = res / 255
                else:
                    res = (res - res.min()) / (res.max() - res.min())

                mae.step(res, gt)
            MAE_ = mae.get_results()['mae']
    model.train()
    return MAE_, validation_step

