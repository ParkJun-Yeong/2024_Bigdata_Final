import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss, test_single_volume, BoundaryDoULoss
from torchvision import transforms
from icecream import ic
import wandb
# from utils import test_single_volume



def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice



def trainer_vfss(args, model, snapshot_path, multimask_output, low_res, valid=True, split='train'):
    from datasets.dataset_vfss import Vfss_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    
    # max_iterations = args.max_iterations
    db_train = Vfss_dataset(base_dir=args.root_path, list_dir=args.list_dir, split=split,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    if valid:
        valid_save_path = "/mnt/ssd01_250gb/juny/vfss/SAMed/output/predictions/valid"
        db_valid = Vfss_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid",
                               transform=False)
        validloader = DataLoader(db_valid, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    if args.loss_fn == "dou":
        boundary_loss = BoundaryDoULoss(num_classes + 1)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            image_batch = image_batch.squeeze(1)            # when not mri/ct image.
            low_res_label_batch = low_res_label_batch.cuda()
            # assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'            # float에서만 해당되는 조항
            outputs = model(image_batch, multimask_output, args.img_size)
            
            if args.loss_fn == "dice_ce":
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            elif args.loss_fn == "dou":
                loss = boundary_loss(outputs['low_res_logits'], low_res_label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            
            if args.loss_fn == "dice_ce":
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/loss_ce', loss_ce, iter_num)
                writer.add_scalar('info/loss_dice', loss_dice, iter_num)

                wandb.log({'total_loss/iter': loss,
                    'loss_ce/iter': loss_ce,
                    'loss_dice/iter': loss_dice,
                    'lr/iter': lr_})

                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            elif args.loss_fn == "dou":
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/boundary_dou_loss', loss, iter_num)

                wandb.log({'total_loss/iter': loss,
                    'lr/iter': lr_})

                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 100 == 0:
                image = image_batch[1, :, :, :]
                image = (image - image.min()) / (image.max() - image.min()) * 255
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...], iter_num)
                labs = label_batch[1, ...].unsqueeze(0)
                writer.add_image('train/GroundTruth', labs, iter_num)

                wandb.log({
                    'train image/per 20 iter': wandb.Image(image.to(torch.float)),
                    'train prediction/per 20 iter': wandb.Image(output_masks[1, ...].to(torch.float)),
                    'train groundtruth/per 20 iter': wandb.Image(labs.to(torch.uint8)),
                })
    

        
        save_interval = int(max_epoch/6) # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            if valid:
                print("=== validation starts ===")

                model.eval()
                metric_list = 0.0
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(validloader):
                        h, w = sampled_batch['image'].shape[2:]
                        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
                        input_size = 224
                        metric_i = test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                                    patch_size=[args.img_size, args.img_size], input_size=[input_size, input_size],
                                                    test_save_path=valid_save_path, case=case_name, z_spacing=1, split="valid")
                        metric_list += np.array(metric_i)
                        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
                        #     i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
                    metric_list = metric_list / len(db_valid)
                    # for i in range(1, args.num_classes + 1):
                    #     try:
                    #         # logging.info('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1]))
                    #         # logging.info('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
                    #     except:
                    #         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
                    performance = np.mean(metric_list, axis=0)[0]
                    mean_hd95 = np.mean(metric_list, axis=0)[1]
                    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
                    logging.info("Testing Finished!")
                    wandb.log({'mean_dice/valid': performance,
                            'mean_hd95/valid': mean_hd95})

            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
