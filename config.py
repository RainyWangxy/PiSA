import os
import sys
import argparse

# from munch import Munch as mch
from os.path import join as ospj


_DATASET = ('pascal', 'coco', 'nuswide', 'cub','vg')
_TRAIN_SET_VARIANT = ('observed', 'clean')
_OPTIMIZER = ('adam', 'sgd')
_Setting = ('missing', 'single','random')
_LOOKUP = {
    'feat_dim': {
        'resnet50': 2048
    },
    'num_classes': {
        'pascal': 20,
        'coco': 80,
        'nuswide': 81,
        'cub': 312,
        'vg':200
    }
}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_dir(runs_dir, exp_name):
    runs_dir = ospj(runs_dir, exp_name)
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    return runs_dir



def set_follow_up_configs(args):
    args.feat_dim = _LOOKUP['feat_dim'][args.arch]
    args.num_classes = _LOOKUP['num_classes'][args.dataset]
    args.save_path = set_dir(args.save_path, args.exp_name)
    return args


def get_configs():
    parser = argparse.ArgumentParser()
    
    ## add settings
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    
    #  Patch settings
    parser.add_argument('--seg', type=str2bool, default=False)
    parser.add_argument('--img_size', default=448, type=int,
                        help='size of input images')
    parser.add_argument("--amp",type=str2bool,default=True)
    parser.add_argument("--mode",type=str,default="train_resnet")
    # parser.add_argument("--loss", type=str,default="bce")
    parser.add_argument("--left_rate", type=float, default=0.5)
    
    parser.add_argument("--freeze_feature_extractor",type=bool,default=False)
    
    # Default settings
    parser.add_argument('--ss_seed', type=int, default=999,
                        help='seed for subsampling')
    parser.add_argument('--ss_frac_train', type=float, default=1.0,
                        help='fraction of training set to subsample')
    parser.add_argument('--ss_frac_val', type=float, default=1.0,
                        help='fraction of val set to subsample')
    parser.add_argument('--use_feats', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='False if end-to-end training, True if linear training')
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--split_seed', type=int, default=1200)
    parser.add_argument('--train_set_variant', type=str, default='observed',
                        choices=_TRAIN_SET_VARIANT)
    parser.add_argument('--val_set_variant', type=str, default='clean')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--num_epochs', type=int, default=10)
    # Util
    parser.add_argument('--save_path', type=str, default='../results/test',)
    
    parser.add_argument('--exp_name', type=str, default='PiSA')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_num', type=str, default='0')
    
    # Data
    
    parser.add_argument('--dataset', type=str, default="pascal",
                        choices=_DATASET)
    parser.add_argument('--pytorch_seed', type=int, default=0)
    # Hyperparameters
    parser.add_argument('--bsize', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--backbone', type=str, default="resnet50")
    parser.add_argument("--dim-embed",type=int,default=2048)
    parser.add_argument('--setting', type=str, default='single',
                        choices=_Setting)
    args = parser.parse_args()
    args = set_follow_up_configs(args)
    return args


