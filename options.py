import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=150, help='epoch number')
parser.add_argument('--model_name', type=str, default="LFNet", help='Model name')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load_mit', type=str, default=r'.\pretrained_params\pvt_v2_b2.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

parser.add_argument('--rgb_root', type=str, default='../dataset/TrainingSet/allfocus/', help='the training rgb images root')
parser.add_argument('--fs_root', type=str, default='../dataset/TrainingSet/focalstack_mat/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='../dataset/TrainingSet/GT/', help='the training gt images root')

# parser.add_argument('--rgb_root', type=str, default='../dataset/New/DUTLF-FS/TrainingSet/allfocus/', help='the training rgb images root')
# parser.add_argument('--fs_root', type=str, default='../dataset/New/DUTLF-FS/TrainingSet/focalstack_mat/', help='the training depth images root')
# parser.add_argument('--gt_root', type=str, default='../dataset/New/DUTLF-FS/TrainingSet/GT/', help='the training gt images root')
#
# parser.add_argument('--rgb_root', type=str, default='../dataset/New/HFUT-Lytro/TrainSet/Allfocus/', help='the training rgb images root')
# parser.add_argument('--fs_root', type=str, default='../dataset/New/HFUT-Lytro/TrainSet/Focalstack_mat/', help='the training depth images root')
# parser.add_argument('--gt_root', type=str, default='../dataset/New/HFUT-Lytro/TrainSet/GT/', help='the training gt images root')
#

parser.add_argument('--test_rgb_root', type=str, default='../dataset/LFSD/allfocus/', help='the test1 gt images root')
parser.add_argument('--test_fs_root', type=str, default='../dataset/LFSD/focalstack_mat/', help='the test1 gt images root')
parser.add_argument('--test_gt_root', type=str, default='../dataset/LFSD/GT/', help='the test1 gt images root')

# parser.add_argument('--test_rgb_root', type=str, default='../dataset/HFUT-Lytro/TestSet/Allfocus/', help='the test1 gt images root')
# parser.add_argument('--test_fs_root', type=str, default='../dataset/New/HFUT-Lytro/TestSet/Focalstack_mat/', help='the test1 gt images root')
# parser.add_argument('--test_gt_root', type=str, default='../dataset/New/HFUT-Lytro/TestSet/GT/', help='the test1 gt images root')

# parser.add_argument('--test_rgb_root', type=str, default='../dataset/New/DUTLF-FS/TestSet/allfocus/', help='the test1 gt images root')
# parser.add_argument('--test_fs_root', type=str, default='../dataset/New/DUTLF-FS/TestSet/focalstack_mat/', help='the test1 gt images root')
# parser.add_argument('--test_gt_root', type=str, default='../dataset/New/DUTLF-FS/TestSet/GT/', help='the test1 gt images root')

parser.add_argument('--save_path', type=str, default='./lfsod_cpts_HFUT/', help='the path to save models and logs')
parser.add_argument("--save_sal", type=str, default="./salient_map/test1/", help="the path to save saliency maps")
opt = parser.parse_args()

