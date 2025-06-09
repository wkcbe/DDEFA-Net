import logging
import os
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from data import test_dataset
from Model.DDFEANet import model
from options import opt
from utils import adjust_lr
from tools.pytorch_utils import Save_Handle
import time
import cv2

torch.cuda.current_device()
print("GPU available:", torch.cuda.is_available())

if opt.gpu_id == '0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('USE GPU 1')
cudnn.benchmark = True

save_list = Save_Handle(max_num=1)

def print_network(model, name):
    """打印神经网络的结构和参数数量：model表示神经网络模型，name表示神经网络的名称。"""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()#numel用于返回数组中的元素个数
    print(name)
    print('The number of parameters:{}'.format(num_params))
    return num_params

test_rgb_root = opt.test_rgb_root
test_fs_root = opt.test_fs_root
test_gt_root = opt.test_gt_root

# 保存路径
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 加载数据
print('load data...')
test_loader = test_dataset(test_rgb_root, test_gt_root, test_fs_root, testsize=opt.trainsize)
step = 0
best_mae = 1
best_epoch = 0

def test(test_loader, model):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, focal, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            dim, height, width = focal.size()
            basize = 1

            focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
            focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
            focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
            focal = focal.view(-1, *focal.shape[2:])
            focal = focal.cuda()
            image = image.cuda()

            _, _, _, _, res, _, _, _, _ = model(focal, image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            # print('save img to: ', save_sal + name)
            # cv2.imwrite(save_sal + name.split('.')[0] + '.png', res * 255)
            cv2.imwrite(save_sal + name, res * 255)


if __name__ == '__main__':
    logging.info("Start train...")
    start_epoch = 0
    model = model()
    # 模型权重加载
    # lfsod_cpts_path = 'E:/DATASET/TestResult/Ours/HFUT/lfsod_epoch_50.pth'
    # lfsod_cpts_path = 'E:/DATASET/TestResult/Ours/LFFS/lfsod_epoch_50.pth'
    lfsod_cpts_path = 'E:/DATASET/TestResult/Ours/LFSD/lfsod_epoch_24.pth'
    model.load_state_dict(torch.load(lfsod_cpts_path))
    model.cuda()
    params = model.parameters()  # 获取一个model的所有参数
    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)  # weight_decay正则化系数
    # 显著图保存地址
    # save_sal = "E:/DATASET/TestResult/Ours/HFUT/lfsod_epoch_50/"
    # save_sal = "E:/DATASET/TestResult/Ours/LFFS/lfsod_epoch_50/"
    save_sal = "E:/DATASET/TestResult/Ours/LFSD/lfsod_epoch_24/"
    if not os.path.exists(save_sal):
        os.makedirs(save_sal)
    start_time = time.time()
    test(test_loader, model)
    end_time = time.time()
    print('Test epoch cost time: {}'.format(end_time - start_time))





































