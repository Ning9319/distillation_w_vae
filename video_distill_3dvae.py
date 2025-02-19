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
from einops import rearrange

warnings.filterwarnings("ignore", category=DeprecationWarning)

import distill_utils

def main(args):

    torch.cuda.set_device(0)  # Ensure it uses the correct device
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    from models.modeling_vae import CVVAEModel

    if args.outer_loop is None and args.inner_loop is None:
        args.outer_loop, args.inner_loop = get_loops(args.ipc)
    elif args.outer_loop is None or args.inner_loop is None:
        raise ValueError(f"Please set neither or both outer/inner_loop: {args.outer_loop}, {args.inner_loop}")
    print('outer_loop = %d, inner_loop = %d'%(args.outer_loop, args.inner_loop))

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    print('Evaluation iterations: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader= get_dataset(args.dataset, args.data_path)

    """
    #Change to only train on the 50% of the original train dataset
    total_samples = len(dst_train)
    subset_size = int(0.5 * total_samples)
    random_indices = random.sample(range(total_samples), subset_size)
    dst_train = Subset(dst_train, random_indices)
    """

    

    
    print("Preloading dataset")
    video_all = []
    label_all = []
    for i in trange(len(dst_train)):
        _ = dst_train[i]
        video_all.append(_[0])
        label_all.append(_[1])
    
    video_all = torch.stack(video_all)
    label_all = torch.tensor(label_all)

    vae3d = CVVAEModel.from_pretrained(args.vae_path, subfolder="vae3d", torch_dtype=torch.float16).to(args.device) # float16 for matching the real_video in the latent space
    vae3d.requires_grad_(False)  # Freeze VAE weights

    #Encode all the videos into the latent space
    video_all = rearrange(video_all, 'b t c h w -> b c t h w').half()
    video_all= video_all/ 127.5 - 1.0
    encode_batch_size = 4
    num_batches = len(video_all) // encode_batch_size + (1 if len(video_all) % encode_batch_size > 0 else 0)
    video_latent = []
    print("\nEncoding the real videos into the latent space\n")
    for i in trange(num_batches):
        batch = video_all[i*encode_batch_size : (i+1)*encode_batch_size]
        batch = batch.to(args.device)
        latents = vae3d.encode(batch).latent_dist.sample()
        video_latent.append(latents) 
    video_all = torch.cat(video_latent, dim=0)

    video_all = rearrange(video_all, 'b c t h w -> b t c h w')
    video_all = video_all.to(torch.float32)
    print("The tensor in the latent space with size:", video_all.shape)  # [B, T, C, H, W]
    dst_train = torch.utils.data.TensorDataset(video_all, label_all)
    

    #syn_video, _ = dst_train.tensors
    #print("Size of the video_all", syn_video.shape)
    




    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    project_name = "Latent_Video_{}".format(args.method)

    
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
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if n == 1:  
            imgs = dst_train[idx_shuffle[0]][0].unsqueeze(0)
        else:
            imgs = torch.cat([dst_train[i][0].unsqueeze(0) for i in idx_shuffle], 0)
        return imgs.to(args.device)
    
    
    latent_im_size = [video_all.shape[-2], video_all.shape[-1]]
    image_syn = torch.randn(size=(num_classes*args.ipc, video_all.shape[-4], video_all.shape[-3], latent_im_size[0], latent_im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)

    label_syn = torch.tensor(np.stack([np.ones(args.ipc)*i for i in range(0, num_classes)]), dtype=torch.long, requires_grad=False,device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    #syn_lr = torch.tensor(args.lr_teacher).to(args.device) if args.method == 'MTT' else None

    if args.init == 'real':
        print('initialize synthetic data from random real images in the latent space')
        for c in range(0, num_classes):
            i = c 
            image_syn.data[i*args.ipc:(i+1)*args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')
 

    
    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True) # [B, T, C, H, W]
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()


    print('%s training begins'%get_time())

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    

    if args.method == "DM":
        for it in trange(0, args.Iteration+1, ncols=60):
            ''' Evaluate synthetic data '''
            
            if it in eval_it_pool:
                save_this_best_ckpt = False
                for model_eval in model_eval_pool:
                    print('Evaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    accs_test = []
                    accs_train = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                        image_syn_eval, label_syn_eval = image_syn.detach().clone(), label_syn.detach().clone() # avoid any unaware modification
                
                        # Applying the decoder to the image_syn_eval
                        image_syn_eval = image_syn_eval.half()
                        image_syn_eval = rearrange(image_syn_eval, 'b t c h w -> b c t h w') # [B, C, T, H, W]
                        reconstructed_videos = []
                        decode_num_batches = len(image_syn_eval) // encode_batch_size + (1 if len(image_syn_eval) % encode_batch_size > 0 else 0)
                        for i in trange(decode_num_batches):
                            batch = image_syn_eval[i * encode_batch_size : (i + 1) * encode_batch_size]
                            batch = batch.to(args.device)
                            decoded_batch  = vae3d.decode(batch).sample
                            reconstructed_videos.append(decoded_batch)
                        image_syn_eval = torch.cat(reconstructed_videos, dim=0)

                        image_syn_eval = (torch.clamp(image_syn_eval,-1.0,1.0) + 1.0) * 127.5
                        image_syn_eval = rearrange(image_syn_eval, 'b c t h w -> b t c h w')
                        
                        print("\nThe image_syn_eval has size of", image_syn_eval.data.shape, "\n") # [B, T, C, H, W]

                        
                        _, acc_train, acc_test, acc_per_cls = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, mode='none',test_freq=100)

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
            
            if it in eval_it_pool and (save_this_best_ckpt or it % 1000 == 0):
                image_save = image_syn.detach()
                save_dir = os.path.join(args.save_path, project_name, wandb.run.name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                if save_this_best_ckpt:
                    save_this_best_ckpt = False
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))

            """
            net = get_network(args.model, video_all.shape[-3], num_classes, latent_im_size, frames=video_all.shape[-4]).to(args.device)  # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if args.distributed else net.embed
            """

            loss_avg = 0

            loss = torch.tensor(0.0).to(args.device)
            for c in range(0,num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, video_all.shape[-4], video_all.shape[-3], latent_im_size[0], latent_im_size[1]))

                # output_real = embed(img_real).detach()
                # output_syn = embed(img_syn)

                #loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                loss += torch.sum((torch.mean(img_real, dim=0) - torch.mean(img_syn, dim=0))**2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()

            loss_avg /= (num_classes)
            

            wandb.log({"Loss": loss_avg}, step=it)
   

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

    parser.add_argument('--vae_path', type=str, default="./vae_weights/3d_vae", help="iterations of training the 3d-vae")




    args = parser.parse_args()

    main(args)

