# TAWNet: Three-dimensional Adaptive Weighted Network for RGB-D Salient Object Detection

## Environment
Please refer to the `requirements.txt` file for the environment configuration we used.

## Datasets
The training and testing datasets for RGBD and RGBT can be downloaded from the following links
[RGB-D datasets](https://github.com/Xiaoqi-Zhao-DLUT/SSLSOD) .

The validation set is a randomly selected image from the test datasets
[BaiduYun](https://pan.baidu.com/s/1duss2wT8Iw3Az_plS3wEYQ)[kbs1] .

Your `/dataset` folder should look like this:

````
-- datasets
   |-- RGB-D_train
   |   |--RGB
   |   |--GT
   |   |--depth
   |-- RGB-D_test
   |   |--DES
   |   |   |--RGB
   |   |   |--GT
   |   |   |--depth
   |   |--NJU2K
       ...
   |-- validation
   |   |--RGB
   |   |--GT
   |   |--depth
````
## RGB-D Saliency map 
The saliency map results on six test datasets of our TAWNet.[BaiduYun](https://pan.baidu.com/s/13yGXEi6DJUjwTqjo4Hpueg)[kbs1]

## Training and Testing
1. Download the pretrained backbone weights and put it into `pretrained_model/` folder. [P2T-base](https://github.com/yuhuan-wu/P2T)

2. Run `python train.py` for training and `python test_eval/test_pred.py` for testing. The trained models will be in `/ckpt` folder. The predictions will be in `test_eval/pred` folder and the training records will be in `results/tensorboard_log/TAWNet_log` folder. 

## Evaluation
Run `python test_eval/eval_pred.py` to obtain the evaluation results of a certain experiment.
We use SOD evaluation tool [PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit) to compare our method with other state-of-the-art methods and plot PR curves.
