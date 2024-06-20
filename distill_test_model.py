import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn

from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, get_attention
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
import kornia as K
import torch.distributed as dist
import torch.cuda.comm
from torchvision.utils import save_image

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') 
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1800, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1, help='learning rate for updating synthetic images, 1 for low IPCs 10 for >= 100')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=64, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real/smart: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='', help='dataset path')
    parser.add_argument('--zca', type=bool, default=False, help='Zca Whitening')
    parser.add_argument('--save_path', type=str, default='', help='path to save results')
    parser.add_argument('--task_balance', type=float, default=0.01, help='balance attention with output')
    
    args = parser.parse_args()
    args.method = 'DataDAM'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    args.save_path += "/{}".format(args.dataset.lower())
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, zca = get_dataset(args.dataset, args.data_path, args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    model_eval = model_eval_pool[0]

    data_save = torch.load(os.path.join(args.save_path, 'syn_data_%s_ipc_%d.pt'%(args.dataset.lower(), args.ipc)))["data"]

    image_syn_eval = torch.tensor(data_save[0])
    label_syn_eval = torch.tensor(data_save[1])
    net_model_dict = torch.load(os.path.join(args.save_path, 'model_params_%s_ipc_%d.pt'%(args.dataset.lower(), args.ipc)))["net_parameters"]
    
    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

    net_eval.load_state_dict(net_model_dict) # load the state dict
    _, _, acc_test = evaluate_synset(-1, net_eval, image_syn_eval, label_syn_eval, testloader, args, skip=True) # evaluate the model
    print("Trained Model Best", acc_test)

main()


