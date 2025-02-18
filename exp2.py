import datetime
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm, trange
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, TensorDataset, epoch, get_loops, match_loss, ParamDiffAug, Conv3DNet
import wandb
import copy
import random
from reparam_module import ReparamModule
import warnings
from torch.utils.data import Subset
import torch.optim as optim

def main(args):

    torch.cuda.set_device(0)  # Ensure it uses the correct device
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    print('Evaluation iterations: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader= get_dataset(args.dataset, args.data_path)

    
    
    print("Preloading dataset")
    video_all = []
    label_all = []
    for i in trange(len(dst_train)):
        _ = dst_train[i]
        video_all.append(_[0])
        label_all.append(_[1])
    
    video_all = torch.stack(video_all)
    label_all = torch.tensor(label_all)
    dst_train = torch.utils.data.TensorDataset(video_all, label_all)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    project_name = "Latent_exp_2"

    wandb.init(sync_tensorboard=False,
            project=project_name,
            job_type="CleanRepo",
            config=args,
            name = f'{args.dataset}_ipc{args.ipc}_{args.lr_img}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
            )

    args = type('', (), {})()


    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    labels_all = label_all if args.preload else dst_train.labels
    indices_class = [[] for c in range(num_classes)]

    print("BUILDING DATASET")
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}


    if args.method == "DM":
        for it in trange(0, args.Iteration+1, ncols=60):
            ''' Evaluate '''
            if it in eval_it_pool:
                save_this_best_ckpt = False
                for model_eval in model_eval_pool:
                    print('Evaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    accs_test = []
                    accs_train = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                        image_syn_eval, label_syn_eval = video_all.detach().clone(), labels_all.detach().clone() # avoid any unaware modification
        
                        
                        print("\nThe image_syn_eval has size of", image_syn_eval.data.shape, "\n") # [B, T, C, H, W]

                        
                        _, acc_train, acc_test, acc_per_cls = evaluate_synset(it_eval, net_eval, image_syn_eval.to(args.device), label_syn_eval.to(args.device), testloader, args, mode='none',test_freq=100)

                        accs_test.append(acc_test)
                        accs_train.append(acc_train)
                        print("acc_per_cls:",acc_per_cls)
                    accs_test = np.array(accs_test)
                    accs_train = np.array(accs_train)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_best_ckpt = True
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                        len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                    wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                    wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                    wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)
            
    wandb.finish()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset')

    parser.add_argument('--method', type=str, default='DC', help='MTT or DM')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='use top5 to eval top5 accuracy, use S to eval single accuracy')
    
    parser.add_argument('--outer_loop', type=int, default=None, help='')
    parser.add_argument('--inner_loop', type=int, default=None, help='')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='how many distillation steps to perform')

    parser.add_argument('--lr_net', type=float, default=0.001, help='learning rate for network')
    parser.add_argument('--lr_img', type=float, default=1, help='learning rate for synthetic data')
    parser.add_argument('--lr_lr', type=float, default=1e-5, help='learning rate for synthetic data')
    parser.add_argument('--lr_teacher', type=float, default=0.001, help='learning rate for teacher')
    parser.add_argument('--train_lr', action='store_true', help='train synthetic lr')
    
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn')

    parser.add_argument('--init', type=str, default='real', choices=['noise', 'real', 'real-all'], help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--data_path', type=str, default='distill_utils/data', help='dataset path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=64, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--buffer_path', type=str, default=None, help='buffer path')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--preload', action='store_true', help='preload dataset')
    parser.add_argument('--save_path',type=str, default='./logged_files', help='path to save')
    parser.add_argument('--frames', type=int, default=16, help='')

    parser.add_argument('--vae_path', type=str, default="./vae_weights/2d_vae", help="iterations of training the 3d-vae")




    args = parser.parse_args()

    main(args)