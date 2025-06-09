import logging
import os
from datetime import datetime
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from data import get_loader, test_dataset
from DDFEANet import model
from options import opt
from utils import clip_gradient, adjust_lr
from tools.pytorch_utils import Save_Handle
from torch.autograd import Variable
from tqdm import tqdm
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
        num_params += p.numel()  # numel用于返回数组中的元素个数
    print(name)
    print('The number of parameters:{}'.format(num_params))
    return num_params
start_epoch = 0

model = model()
# 模型权重加载
# Model.load_state_dict(torch.load('E:\my_project\LFSOD\LFTransNet-test2\lfsod_cpts\lfsod_epoch_147_1.pth'))
if (opt.load_mit is not None):
    model.focal_encoder.init_weights(opt.load_mit)
    model.rgb_encoder.init_weights(opt.load_mit)
else:
    print("No pre-trian!")

model.cuda()
params = model.parameters()  # 获取一个model的所有参数
optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)  #weight_decay正则化系数
# 打印模型信息
model_params = print_network(model, 'lf_pvt')
# 传入train+test1 数据地址
rgb_root = opt.rgb_root
gt_root = opt.gt_root
fs_root = opt.fs_root
test_rgb_root = opt.test_rgb_root
test_fs_root = opt.test_fs_root
test_gt_root = opt.test_gt_root
save_sal = opt.save_sal
if not os.path.exists(save_sal):
    os.makedirs(save_sal)
# 保存路径
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)
# 加载数据
print('load data...')
train_loader = get_loader(rgb_root, gt_root, fs_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_rgb_root, test_gt_root, test_fs_root,testsize=opt.trainsize)
total_step = len(train_loader)
logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Net-Train")
logging.info("Config")
logging.info('params:{}'.format(model_params))
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate,  save_path,
        opt.decay_epoch))


def structure_loss(pred, mask):
    # 损失函数
    weit = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


step = 0
best_mae = 1
best_epoch = 0

def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    start_time = time.time()  # 记录整个epoch的开始时间
    with tqdm(total=len(train_loader), desc='Epoch {}'.format(epoch), ncols=80) as pbar:
        try:
            for i, (images, gts, focal) in enumerate(train_loader, start=1):
                basize, dim, height, width = focal.size()
                gts = gts.cuda()
                images, gts, focal = Variable(images), Variable(gts), Variable(focal)
                # 获得不同stage的显著图GT
                gts1 = F.interpolate(gts, size=(64, 64), mode='bilinear', align_corners=False)  # (1,1,64,64)
                gts2 = F.interpolate(gts, size=(32, 32), mode='bilinear', align_corners=False)  # (1,1,32,32)
                gts3 = F.interpolate(gts, size=(16, 16), mode='bilinear', align_corners=False)  # (1,1,16,16)
                gts4 = F.interpolate(gts, size=(8, 8), mode='bilinear', align_corners=False)    # (1,1,8,8)
                # 焦堆栈图像读取
                focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
                focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
                focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 3, 256, 256)
                focal = focal.view(-1, *focal.shape[2:])  # [basize*12, 3, 256, 256)
                focal = focal.cuda()  # (12,3,256,256)
                images = images.cuda()  # (1,3,256,256)
                optimizer.zero_grad()
                x1, x2, x3, x4, focal_sal, _, _, _, _ = model(focal, images)
                # focal_sal:(1,1,256,256),x1_4:{1,1,(64,32,16,8)}
                # 计算损失并反向传播,gts:(1,1,256,256)
                loss = structure_loss(focal_sal, gts) + structure_loss(x1, gts1) + structure_loss(x2, gts2) + structure_loss(x3, gts3) + structure_loss(x4, gts4)
                loss.backward()
                # 梯度裁剪
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                step += 1
                epoch_step += 1
                loss_all += loss.data
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                if i % 100 == 0 or i == total_step or i == 1:
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                          format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                    logging.info(
                        '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} ||Mem_use:{:.0f}MB'.
                            format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data, memory_used))
                # 更新进度条
                pbar.set_description(
                    'Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(epoch, opt.epoch, i,len(train_loader),loss.item()))
                pbar.update(1)
                if i % 100 == 0 or i == len(train_loader) or i == 1:
                    curr_time = time.time()
                    elapsed_time = curr_time - start_time
                    print('Time Elapsed: {:.2f}s, Estimated Remaining Time: {:.2f}s'.format(elapsed_time, (
                                len(train_loader) - i) * (elapsed_time / i)))
                    start_time = curr_time  # 重置开始时间以计算下一轮的运行时间
            loss_all /= epoch_step
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
            if epoch % 2 == 0:
                torch.save(model.state_dict(), save_path + 'lfsod_epoch_{}.pth'.format(epoch))

            # 训练中断保留参数
            temp_save_path = save_path + "{}_ckpt.tar".format(epoch)
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()
            }, temp_save_path)
            save_list.append(temp_save_path)

        except KeyboardInterrupt:
            logging.info('Keyboard Interrupt: save Model and exit.')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + 'lfnet_epoch_{}.pth'.format(epoch + 1))
            logging.info('save checkpoints successfully!')
            raise


if __name__ == '__main__':
    logging.info("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    for epoch in range(start_epoch, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, save_path)






































