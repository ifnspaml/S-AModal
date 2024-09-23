# this script is our main script for fine-tuning SAM to the amodal task
# before running this script make sure you have downloaded the necessary SAM checkpoints
# this script is build upon the AutoSAM training script: https://github.com/talshaharabany/AutoSAM/blob/main/train.py

import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np

import sys
sys.path.append('./datasets/')
from datasets.amodal_synth_drive import get_asd_dataset
from datasets.kins_car import get_kins_car_dataset

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
import torch.distributed as dist
from utils.helper_functions import *

local_rank = 0

print('local_rank is', local_rank)


global GPU


GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.8
        self.gamma = 2

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        exp_val = torch.exp(-bce)
        focal = self.alpha * (1 - exp_val)**self.gamma * bce

        return focal


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1) / \
            (inputs.sum() + targets.sum() + 1)
        return 1 - dice


def criterion(x, y):
    """ Combined dice and focal loss.
    ARGS:
        x: (torch.Tensor) the model output
        y: (torch.Tensor) the target
    RETURNS:
        (torch.Tensor) the combined loss

    """
    focal, dice = FocalLoss(), DiceLoss()
    y = y.to(GPU)
    x = x.to(GPU)
    return 20 * focal(x, y) + dice(x, y)


# the following code is adapted from AutoSAM (https://github.com/talshaharabany/AutoSAM/blob/main/train.py)
def inference_ds(ds, sam, transform, epoch, args):
    pbar = tqdm(ds)
    sam.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])

    for ix, (imgs, gts, original_sz, img_sz, point) in enumerate(pbar):
        if ix*4>500:
            break

        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)

        with torch.no_grad():
            image_embedding = sam.image_encoder(orig_imgs)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=point,
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        upscaled_masks = sam.postprocess_masks(low_res_masks, (img_sz[0][0], img_sz[1][0]), (original_sz[0][0],original_sz[1][0])).to(sam.device)


        from torch.nn.functional import threshold, normalize

        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(sam.device)
        iou = get_iou(binary_mask.squeeze().detach().cpu().numpy(),
                               gts.squeeze().detach().cpu().numpy())
        iou_list.append(iou)
        if ix % 50 == 0:
            pbar.set_description(
                '(Inference | {task}) Epoch {epoch} :: IoU {iou:.4f}'.format(
                    task=args['task'],
                    epoch=epoch,
                    iou=np.mean(iou_list)))
    sam.train()
    return np.mean(iou_list)

def apply_coords_torch_reformulate(
    coords, original_size):
    """
    Expects a torch tensor with length 2 in the last dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size

    new_h, new_w = 1024,1024

    coords = deepcopy(coords).to(torch.float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

# the following function is adapted from https://github.com/talshaharabany/AutoSAM/blob/main/train.py
def sam_call(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks

# the following function is adapted from https://github.com/talshaharabany/AutoSAM/blob/main/train.py
def main(args=None, sam_args=None):
    sam = sam_model_registry[sam_args['model_type']](args,checkpoint=sam_args['sam_checkpoint'])
    device= torch.device('cuda', sam_args['gpu_id'])
    sam.to(device=device)
    print('model loaded', flush=True)


    transform = ResizeLongestSide(sam.image_encoder.img_size)

    if 'adam' in args['optimizer']:
        optimizer = torch.optim.Adam(sam.mask_decoder.parameters(),lr=float(args['learning_rate']),
                               weight_decay=float(args['WD']))
    elif 'aw' in args['optimizer']:
        optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(),lr=float(args['learning_rate']),
                               weight_decay=float(args['WD']))
    #try galore optimizer
    elif 'galore' in args['optimizer']:
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in sam.mask_decoder.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)

        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        non_galore_params = [p for p in sam.mask_decoder.parameters() if id(p) not in id_galore_params]



        param_groups = [{'params': non_galore_params},
                        {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}]
        optimizer = GaLoreAdamW(param_groups, lr=0.01)

    # decide on dataset
    if 'asd' in args['dataset']:
        trainset, testset = get_asd_dataset(sam, sam_trans=transform)
    elif 'kcar' in args['dataset']:
        trainset, testset = get_kins_car_dataset(sam, sam_trans=transform)


    ds = torch.utils.data.DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                                     num_workers=int(args['nW']), drop_last=True)
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    best = 0
    path_best = 'results/' + args['full_pathname']  + '/best.csv'
    f_best = open(path_best, 'w')
    max_iterations = args['epoches'] * len(ds)
    if 'samadpt' in args['mode']:
        for name, para in sam.named_parameters():
            if "image_encoder" in name and "prompt_generator" not in name:
                para.requires_grad_(False)
        if local_rank == 0:
            model_total_params = sum(p.numel() for p in sam.parameters())
            model_grad_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
            print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
    else:
        for name, para in sam.named_parameters():
            if "image_encoder" in name:
                para.requires_grad_(False)
        if local_rank == 0:
            model_total_params = sum(p.numel() for p in sam.parameters())
            model_grad_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
            print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    iter_num=0
    for epoch in range(int(args['epoches'])):
        loss_list = []

        optimizer.zero_grad()
        for ix, (imgs, gts, original_sz, img_sz, point) in enumerate(ds):


            orig_imgs = imgs.to(sam.device)
            gts = gts.to(sam.device)


            image_embedding = sam.image_encoder(orig_imgs)

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=point,
                    boxes=None,
                    masks=None,
                )

            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam.postprocess_masks(low_res_masks, (img_sz[0][0],img_sz[1][0]), (original_sz[0][0],original_sz[1][0])) #kins is 375,1242

            from torch.nn.functional import threshold, normalize

            # binary_mask = normalize(threshold(upscaled_masks.cpu().detach(), 0.0, 0))#threshold(upscaled_masks, 0.0, 0).to(device)#

            loss = criterion(upscaled_masks.to(device), gts.float())

            optimizer.zero_grad()
            loss.backward()
            loss_list.append(loss.item())

            if ix%2000==0:
                print('(train | {}) epoch {epoch} ::'
                    ' loss {loss:.4f}'.format(
                        'AmodalSAM',
                        epoch=epoch,
                        loss=np.mean(loss_list)),flush=True)
                #plt.imsave('kins_car_gts_%s.png' % ix, gts[0, 0].float().cpu().detach().numpy())
                #plt.imsave('kins_car_binary_mask_%s.png' % ix, binary_mask[0, 0].numpy())
                #plt.imsave('kins_car_upscaled_masks%s.png' % ix, upscaled_masks[0, 0].cpu().detach().numpy())

            optimizer.step()
            if args['if_warmup'] and iter_num < args['warmup_period']:
                lr_ = float(args['learning_rate']) * ((iter_num + 1) / args['warmup_period'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            else:
                if args['if_warmup']:
                    shift_iter = iter_num - args['warmup_period']
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = float(args['learning_rate']) * (1.0 - shift_iter / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num +=1


        with torch.no_grad():
            IoU_val = inference_ds(ds_val, sam.eval(), transform, epoch, args)
            print('current eval result: ' + str(IoU_val))
            checkpoint = {
                'epoch': epoch,
                'model': sam.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler}
            torch.save(checkpoint, 'results/' + args['full_pathname'] + '/checkpoint_epoch%s.pth' % epoch)
            if IoU_val > best:
                torch.save(sam.state_dict(), args['path_best'])

                best = IoU_val
                print('best results: ' + str(best))
                f_best.write(str(epoch) + ',' + str(best) + '\n')
                f_best.flush()
        lr_scheduler.step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.00001, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=8, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=10, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=5e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='kins', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=1024, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    parser.add_argument('-mid_dim', type=int, default=None, help='middle dim of adapter or the rank of lora matrix')
    parser.add_argument('-mode', type=str, default='normal', help='mode is adapt or normal, samadpt (from https://github.com/tianrun-chen/SAM-Adapter-PyTorch/tree/main)')
    #adapt mode from: https://github.com/KidsWithTokens/Medical-SAM-Adapter/tree/main
    #normal mode is standard sam
    parser.add_argument('-warmup_period', type=int, default=200, help='number of warumup iterations')
    parser.add_argument('-if_warmup', type=bool, default=False,help='whether warmup should be used')

    parser.add_argument('-optimizer', type=str, default='adam', help='current options: adam, galore, aw')
    parser.add_argument('-dataset', type=str, default='kins', help='current options: kins,asd, kcar')
    args = vars(parser.parse_args())
    os.makedirs('results', exist_ok=True)
    folder = open_folder('results')
    args['folder'] = folder
    full_pathname = args['dataset'] + '_' + args['Batch_size'] + '_' + args['learning_rate'] + '_' \
                    + args['optimizer'] + '_' + args['mode'] + '_gpu' + folder
    os.mkdir('results/' + full_pathname)
    args['full_pathname'] = full_pathname
    args['path'] = os.path.join('results',
                                full_pathname ,
                                'net_last.pth')
    args['path_best'] = os.path.join('results',
                                     full_pathname,
                                     'net_best.pth')
    args['vis_folder'] = os.path.join('results', full_pathname, 'vis')
    os.mkdir(args['vis_folder'])
    print('save path is', args['path'])
    sam_args = {
        'sam_checkpoint': "./checkpoints/sam_vit_b_01ec64.pth",
        'model_type': "vit_b",
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }
    main(args=args, sam_args=sam_args)

