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

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist()[:] if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, zca = get_dataset(args.dataset, args.data_path, args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    
    total_mean = {}
    best_5 = []
    accuracy_logging = {"mean":[], "std":[], "max_mean":[]}
    for exp in range(args.num_exp):
        total_mean[exp] = {'mean':[], 'std':[]}
        best_5.append(0)
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)



        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        elif args.init =='noise' :
            print('initialize synthetic data from random noise')
            
        elif args.init =='smart' :
            print('initialize synthetic data from SMART selection')
            Path = './'
            if args.dataset == "CIFAR10":
                Path+='CIFAR10_'
            
            elif args.dataset == "CIFAR100":
                Path+='CIFAR100_'
                
            if args.ipc == 1:
                Path += 'IPC1_'
                
            elif args.ipc == 10:
                Path += 'IPC10_'
                
            elif args.ipc == 50:
                Path += 'IPC50_'
                
            elif args.ipc == 100:
                Path += 'IPC100_'
            
            elif args.ipc == 200:
                Path += 'IPC200_'
            image_syn.data[:][:][:][:] = torch.load(Path+'images.pt')
            label_syn.data[:] = torch.load(Path+'labels.pt')
            
        if(args.zca):
            print("ZCA Whitened Complete")
            image_syn.data[:][:][:][:] = zca(image_syn.data[:][:][:][:], include_fit=True)
        else:
            print("No ZCA Whiteinign")
                
            
            
            
        
        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())
        ''' Defining the Hook Function to collect Activations '''
        activations = {}
        def getActivation(name):
            def hook_func(m, inp, op):
                activations[name] = op.clone()
            return hook_func
        
        ''' Defining the Refresh Function to store Activations and reset Collection '''
        def refreshActivations(activations):
            model_set_activations = [] # Jagged Tensor Creation
            for i in activations.keys():
                model_set_activations.append(activations[i])
            activations = {}
            return activations, model_set_activations
        
        ''' Defining the Delete Hook Function to collect Remove Hooks '''
        def delete_hooks(hooks):
            for i in hooks:
                i.remove()
            return
        
        def attach_hooks(net):
            hooks = []
            base = net.module if torch.cuda.device_count() > 1 else net
            for module in (base.features.named_modules()):
                if isinstance(module[1], nn.ReLU):
                    # Hook the Ouptus of a ReLU Layer
                    hooks.append(base.features[int(module[0])].register_forward_hook(getActivation('ReLU_'+str(len(hooks)))))
            return hooks
        
        max_mean = 0
        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    Start = time.time()
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        mini_net, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                        if acc_test > best_5[-1]:
                            best_5[-1] = acc_test
                    
                    Finish = (time.time() - Start)/10
                    
                    print("TOTAL TIME WAS: ", Finish)
                            
                            
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    if np.mean(accs) > max_mean:
                        data=[]
                        data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                        torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc_.pt'%(args.method, args.dataset, args.model, args.ipc)))
                    # Track All of them!
                    total_mean[exp]['mean'].append(np.mean(accs))
                    total_mean[exp]['std'].append(np.std(accs))
                    
                    accuracy_logging["mean"].append(np.mean(accs))
                    accuracy_logging["std"].append(np.std(accs))
                    accuracy_logging["max_mean"].append(np.max(accs))
                    
                    
                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                # save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                # image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                # for ch in range(channel):
                #     image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                # image_syn_vis[image_syn_vis<0] = 0.0
                # image_syn_vis[image_syn_vis>1] = 1.0
                # save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False
                    
            loss_avg = 0
            def error(real, syn, err_type="MSE"):
                        
                if(err_type == "MSE"):
                    err = torch.sum((torch.mean(real, dim=0) - torch.mean(syn, dim=0))**2)
                
                elif (err_type == "MAE"):
                    err = torch.sum(torch.abs(torch.mean(real, dim=0) - torch.mean(syn, dim=0)))
                    
                elif (err_type == "ANG"):
                    rl = torch.mean(real, dim=0) 
                    sy = torch.mean(syn, dim=0)
                    num = torch.matmul(rl, sy)
                    denom = (torch.sum(rl**2)**0.5) * (torch.sum(sy**2)**0.5)
                    err = torch.acos(num/denom)
                    
                elif(err_type == "MSE_B"):
                    err = torch.sum((torch.mean(real.reshape(num_classes, args.batch_real, -1), dim=1).cpu() - torch.mean(syn.cpu().reshape(num_classes, args.ipc, -1), dim=1))**2)
                elif(err_type == "MAE_B"):
                    err = torch.sum(torch.abs(torch.mean(real.reshape(num_classes, args.batch_real, -1), dim=1).cpu() - torch.mean(syn.reshape(num_classes, args.ipc, -1).cpu(), dim=1)))
                elif (err_type == "ANG_B"):
                    rl = torch.mean(real.reshape(num_classes, args.batch_real, -1), dim=1).cpu()
                    sy = torch.mean(syn.reshape(num_classes, args.ipc, -1), dim=1)
                    
                    denom = (torch.sum(rl**2)**0.5).cpu() * (torch.sum(sy**2)**0.5).cpu()
                    num = rl.cpu() * sy.cpu()
                    err = torch.sum(torch.acos(num/denom))
                return err
            
            ''' update synthetic data '''
            loss = torch.tensor(0.0)
            mid_loss = 0
            out_loss = 0

            images_real_all = []
            images_syn_all = []
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                images_real_all.append(img_real)
                images_syn_all.append(img_syn)

            images_real_all = torch.cat(images_real_all, dim=0)
            
            images_syn_all = torch.cat(images_syn_all, dim=0)

            
            hooks = attach_hooks(net)
            
            output_real = net(images_real_all)[0].detach()
            activations, original_model_set_activations = refreshActivations(activations)
            
            output_syn = net(images_syn_all)[0]
            activations, syn_model_set_activations = refreshActivations(activations)
            delete_hooks(hooks)
            
            length_of_network = len(original_model_set_activations)# of Feature Map Sets
            
            for layer in range(length_of_network-1):
                
                real_attention = get_attention(original_model_set_activations[layer].detach(), param=1, exp=1, norm='l2')
                syn_attention = get_attention(syn_model_set_activations[layer], param=1, exp=1, norm='l2')

                tl =  100*error(real_attention, syn_attention, err_type="MSE_B")
                loss+=tl
                mid_loss += tl

            output_loss =  100*args.task_balance * error(output_real, output_syn, err_type="MSE_B")
            
            loss += output_loss
            out_loss += output_loss

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            torch.cuda.empty_cache()

            loss_avg /= (num_classes)
            out_loss /= (num_classes)
            mid_loss /= (num_classes)
            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))
    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
    
    print('\n==================== Maximum Results ====================\n')
    
    best_means = []
    best_std = []
    for exp in total_mean.keys():
        best_idx = np.argmax(total_mean[exp]['mean'])
        best_means.append(total_mean[exp]['mean'][best_idx])
        best_std.append(total_mean[exp]['std'][best_idx])
    
    mean = np.mean(best_means)
    std = np.mean(best_std)
        
    num_eval = args.num_exp*args.num_eval
    print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model,num_eval, key, mean*100, std*100))
    
    
    print('\n==================== Top 5 Results ====================\n')
    
       
    mean = np.mean(best_5)
    std = np.std(best_5)
        
    num_eval = args.num_exp*args.num_eval
    print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model,num_eval, key, mean*100, std*100))


if __name__ == '__main__':
    main()


