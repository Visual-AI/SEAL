import argparse

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits, get_datasets_v2

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root,dino_pretrain_path, dinov2_pretrain_path
from model import ContrastiveLearningViewGenerator, get_params_groups, vit_threeHeads_v2, vit_twoHeads_v2
import vision_transformer as vits
import vision_transformers_v2 as vits_v2
import matplotlib.pyplot as plt
import gc

from birds_category import trees as birds_category_list
from birds_category import get_order_family_target as get_birds_order_family_target
from aircraft_category import trees as aircraft_category_list
from aircraft_category import get_order_family_target as get_aircraft_order_family_target

from cars_category import trees as cars_category_list
from cars_category import get_order_family_target as get_cars_order_family_target



two_level_datasets = ['scars']


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def test(model, test_loader, epoch, save_name, args, get_order_family_target):

    model.eval()

    preds, targets = [], []

    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
    
        with torch.no_grad():
            (order_proj, order_out), (family_proj, family_out), (species_proj, species_out) = model(images)
            

            
            preds.append(species_out.argmax(1).cpu().numpy())

            
            targets.append(label.cpu().numpy())
            
  
            
            mask = np.append(mask, np.array([True if x.item() in args.train_classes else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
   
    

      
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1','v2', 'v2b'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--memax_weight_1', type=float, default=0.5)
    parser.add_argument('--memax_weight_2', type=float, default=0.5)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    
    parser.add_argument('--random_seed', default=666, type=int)
    parser.add_argument('--model_name', default='vit_dino', type=str)
    
    parser.add_argument('--hyper_start_epoch', default=0, type=int)
    parser.add_argument('--hyper_end_epoch', default=200, type=int)

    parser.add_argument('--feature_size', default=768, type=int)
    parser.add_argument('--similarity', default='cos', type=str)
    parser.add_argument('--update_thd', type=float, default=0.1)
    parser.add_argument('--M_momentum', type=float, default=0.9)
    parser.add_argument('--kl_temp', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    
    parser.add_argument('--sim_alpha', type=float, default=0.6)
    parser.add_argument('--sim_beta', type=float, default=0.3)
    parser.add_argument('--sim_gamma', type=float, default=0.1)
    
    parser.add_argument('--backbone_lr', type=float, default=0.1)
    parser.add_argument('--features_lr', type=float, default=0.1)
    parser.add_argument('--projector_1_lr', type=float, default=0.1)
    parser.add_argument('--projector_2_lr', type=float, default=0.1)
    parser.add_argument('--projector_3_lr', type=float, default=0.1)
    
    parser.add_argument('--unsupervised_smoothing', type=float, default=0.5)
    
    parser.add_argument('--P_momentum', type=float, default=0.9)
    parser.add_argument('--warmup_epoch_matrix', default=30, type=int, help='warmup epoch for matrix momentum update')
    parser.add_argument('--warmup_epoch_prototype', default=30, type=int, help='warmup epoch for matrix momentum update')
    
    parser.add_argument('--save_model_path', type=str)
    

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    print(args)
    set_random_seed(args.random_seed)
    device = torch.device('cuda:0')
    args.device=device
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[f'simgcd_baseline'])
    args.logger.info(f'Using evaluation function {args.eval_funcs} to print results')
    
    # torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768 
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    if args.model_name == 'vit_dino':
        backbone = vits.__dict__['vit_base']()

        state_dict = torch.load(dino_pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
    elif args.model_name == 'vit_dino_v2':
        backbone = vits_v2.__dict__['vit_base']()
        state_dict = torch.load(dinov2_pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
    else:
        raise ValueError('Invalid model name')
    # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

   


    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets, train_dataset_test, labelled_train_examples_test = get_datasets_v2(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    # projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    # model = nn.Sequential(backbone, projector).to(device)
    if args.dataset_name == 'cub':
        num_superclass = max([i[1] for i in birds_category_list])
        num_fine = max([i[2] for i in birds_category_list])
    elif args.dataset_name == 'aircraft':
        num_superclass = max([i[2] for i in aircraft_category_list])
        num_fine = max([i[1] for i in aircraft_category_list])
    elif args.dataset_name  == 'scars':
        num_superclass = max([i[1] for i in cars_category_list])
        num_fine = 0

    else:
        raise ValueError("Not Support for this dataset")
    
    get_order_family_target_dict = {
        'cub': get_birds_order_family_target,
        'aircraft': get_aircraft_order_family_target,
        'scars': get_cars_order_family_target,
    }
    
    if args.dataset_name in two_level_datasets:
        model = vit_twoHeads_v2(backbone=backbone,in_dim= args.feat_dim, num_class=num_fine,num_superclass = num_superclass,num_fine=args.mlp_out_dim, nlayers=args.num_mlp_layers, feature_size = args.feature_size)
    else:
        model = vit_threeHeads_v2(backbone=backbone,in_dim= args.feat_dim, num_class=num_fine,num_superclass = num_superclass,num_fine=args.mlp_out_dim, nlayers=args.num_mlp_layers, feature_size = args.feature_size)
    # model = nn.DataParallel(model) 
    state_dict = torch.load(args.save_model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model = model.cuda()
    
    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    args.num_species = args.num_labeled_classes + args.num_unlabeled_classes
    args.num_families = num_fine
    args.num_orders = num_superclass
    all_acc_test, old_acc_test, new_acc_test = test(model, test_loader_unlabelled, epoch=0, save_name='Test ACC', args=args, get_order_family_target=None)
    print(f'Test Accuracies on Unlabelled Examples: All {all_acc_test:.4f} | Old {old_acc_test:.4f} | New {new_acc_test:.4f}')
