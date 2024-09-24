# this script has a lot of options that need to be set
# make sure to set them before running
# this script is based on the inference script in https://github.com/talshaharabany/AutoSAM/blob/main/inference.py

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from datasets.amodal_synth_drive import ASD_Dataset
from datasets.kins_car import get_kins_car_dataset, KINSCarDataset
from evaluation.ap_evaluation import customCOCOeval as COCOeval
from utils.helper_functions import *
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from tqdm import tqdm

def main(args):
    single_image = False

    device = "cuda"

    # dataset = 'kcar'
    # dataset = 'asd'
    dataset = 'sailvos'

    modeltype = "vit_b"
    model_checkpoint = "/beegfs/work/breitenstein/segment-anything/results/asd_1_0.00001_aw_samadpt_gpu270/net_best.pth"
    # model_checkpoint = "/beegfs/work/breitenstein/segment-anything/results/kcar_1_0.00001_aw_samadpt_gpu235/net_best.pth"


    mode = {'mode': 'samadpt'} #'mode is adapt or normal, samadpt
    pt_augmentation = None  # 'saliency' # # can be: maxdis, maxent, random, saliency
    num_points = 1  # set number of points for prompting SAM
    print('settings are:', flush=True)
    print('dataset: ', dataset, flush=True)
    print('modeltype: ', modeltype, flush=True)
    print('model_checkpoint: ', model_checkpoint, flush=True)
    print('mode: ', mode, flush=True)
    print('pt_augmentation: ', pt_augmentation, flush=True)
    print('num_points: ', num_points, flush=True)
    print('savepath is: ', args['save_path'], flush=True)

    sam = sam_model_registry[modeltype](mode,checkpoint=model_checkpoint)

    sam.to(device=device)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    sam.eval()
    if single_image:

        image = cv2.imread('/beegfs/work/breitenstein/sam-pt/data/amodal_synth_drive/JPEGImages/20230623143953_DustStorm_Town10HD/front_full_0007_rgb.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_shape = (1080, 1920)


        input_point = np.array([[1870., 609.],
                                [1883., 534.],
                                [1830., 562.],
                                [1899., 540.]])

        input_label = np.array([1, 1, 1, 1])


        img = torch.tensor(image)
        print('img shape', img.shape)
        img = img.permute((2, 0, 1))
        img = F.interpolate(img[np.newaxis],size=image_shape,mode='bilinear')

        # Transform the image to the form expected by the model
        input_image = transform.apply_image(np.array(img[0].permute((1,2,0))))
        input_image_torch = torch.as_tensor(input_image, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        original_size = img.shape[:2]
        input_size = tuple(input_image_torch.shape[-2:])
        input_image_preprocess = (input_image_torch - sam.pixel_mean) / sam.pixel_std

        # Pad
        h, w = input_image_preprocess.shape[-2:]
        padh = sam.image_encoder.img_size - h
        padw = sam.image_encoder.img_size - w
        input_image_preprocess = F.pad(input_image_preprocess, (0, padw, 0, padh))


        point_coords = transform.apply_coords(input_point, image_shape)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        input_sam = input_image_preprocess[0]

        input_sam_point =[coords_torch[0], labels_torch[0]]
        print('input sam point is', input_sam_point)
        modified_list = [tensor.unsqueeze(0) for tensor in input_sam_point]# if len(tensor.shape) == 2 else tensor.unsqueeze(0).unsqueeze(0) for tensor in input_sam_point]
        print(input_sam_point)
        print(modified_list)
        print('input_sam shape', input_sam.shape)

        with torch.no_grad():
            image_embedding = sam.image_encoder(input_sam[None])
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=modified_list,
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
        # upscaled_masks = sam.postprocess_masks(low_res_masks, img_sz, (375, 1242)).to(sam.device)
        upscaled_masks = sam.postprocess_masks(low_res_masks, (input_size[0], input_size[1]), image_shape).to(sam.device)

        # upscaled_masks = sam.postprocess_masks(low_res_masks, (Idim, Idim), original_sz).to(sam.device)

        from torch.nn.functional import threshold, normalize

        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(sam.device)
        print('binary_mask shape', binary_mask.shape)
        print('iou_predictions', iou_predictions)
        print('iou_predictions shape', iou_predictions.shape)

        plt.figure(figsize=(10, 10))
        plt.imshow(image.astype('uint8'))#.transpose((1, 2, 0))
        plt.imshow(binary_mask[0,0].cpu().detach().numpy(), alpha=0.5)
        show_points(input_point[0], input_label[0], plt.gca())
        plt.savefig('output.png')
        plt.close()

        overlay_color = [255, 0, 0]  # Red: [R, G, B]
        # Define the transparency level (alpha) for the overlay
        alpha = 0.5  # Adjust as needed, 0 is fully transparent, 1 is fully opaque
        # Create a copy of the original image to overlay
        overlay_image = np.copy(image).astype('uint8')
        # Create a mask for the overlay based on the binary mask
        overlay_mask = np.zeros_like(overlay_image)
        overlay_mask[binary_mask[0,0].cpu().detach().numpy() == 1] = overlay_color  # Set alpha channel to 0 for fully transparent areas
        # Combine the original image and the overlay with transparency
        overlay_image = (1 - alpha) * overlay_image + alpha * overlay_mask
        plt.imsave('overlay_visualization.png', overlay_image.astype('uint8'))
    else:
        #trainset,testset = get_kins_dataset(0,sam, sam_trans=transform, test=True)

        if 'asd' in dataset:
            image_root = "/beegfs/data/shared/amodal_synth_drive_new_split/val/images/front/"
            gt_root = "/beegfs/data/shared/amodal_synth_drive_new_split/val/amodal_instance_seg/front/"
            testset = ASD_Dataset(image_root, gt_root, sam, train=False, sam_trans=transform, test=True,
                                  pt_augmentation=pt_augmentation, num_points= num_points)
            catId = 24
        elif 'kcar' in dataset:
            image_root = "/beegfs/data/shared/kitti-kins/KINS_Video_Car/Kins/training/image_2/"
            gt_root = "/beegfs/data/shared/kitti-kins/KINS_Video_Car/car_data/test_data.pkl"
            testset = KINSCarDataset(image_root, gt_root, sam, train=False, sam_trans=transform, test=True)
            catId = 4




        ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                             num_workers=int(0), drop_last=False)
        pbar = tqdm(ds_val)
        sam.eval()
        iou_list = []
        dice_list = []
        tp=0
        fp=0
        fn=0

        #code to evaluate average precision
        # Step 1: Collect predictions and ground truth annotations
        predictions = []

        for ix, (imgs, gts, original_sz, img_sz, point,vis_img, image_id,
                 visible_mask, origpoint, origpoint_labels, track_id, video_id) in enumerate(pbar):

            point_json_format = [origpoint[0,0,0].item(),origpoint[0,0,1].item()]

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
            #upscaled_masks = sam.postprocess_masks(low_res_masks, img_sz, (375, 1242)).to(sam.device)
            upscaled_masks = sam.postprocess_masks(low_res_masks, (img_sz[0][0],img_sz[1][0]), (original_sz[0][0],original_sz[1][0])).to(sam.device)

            #upscaled_masks = sam.postprocess_masks(low_res_masks, (Idim, Idim), original_sz).to(sam.device)

            from torch.nn.functional import threshold, normalize

            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(sam.device)
            iou = get_iou(binary_mask.squeeze().detach().cpu().numpy(),
                                   gts.squeeze().detach().cpu().numpy())

            predicted_mask = binary_mask.squeeze().detach().cpu().numpy().astype(np.uint8)
            predicted_mask2 = predicted_mask + visible_mask.squeeze().detach().cpu().numpy().astype(np.uint8)
            predicted_mask2[predicted_mask2>1]=1

            predicted_mask = np.asfortranarray(predicted_mask2)

            #combine tracking id with video id
            if 'asd' in dataset:
                new_track_id = str(track_id)+ str(int(image_id/100))
                new_track_id = int(new_track_id)
            elif 'sailvos' in dataset:
                new_track_id = track_id.item()




            pred_mask_encoded = mask_utils.encode(predicted_mask)
            pred_mask_encoded['counts'] = str(pred_mask_encoded['counts'],'utf-8')
            predictions.append({'image_id': image_id.item(),
                                'video_id': video_id.item(),
                                'segmentation': pred_mask_encoded,
                                'score': iou_predictions.item(),
                                'category_id': catId,
                                'height': original_sz[0][0].item(),
                                'width': original_sz[1][0].item(),
                                'track_id': new_track_id,
                                'original_point': point_json_format})
            #set category_id =1 for all predictions to avoid errors in loading with pycocotools

            #predict = binary_mask.squeeze().detach().cpu().numpy() + 1
            predict = binary_mask.squeeze().detach().cpu().numpy() + visible_mask.squeeze().detach().cpu().numpy()
            predict[predict>1]=1
            predict = predict + 1
            target = gts.squeeze().detach().cpu().numpy() + 1
            tp += np.sum(((predict == 2) * (target == 2)) * (target > 0))
            fp += np.sum(((predict == 2) * (target == 1)) * (target > 0))
            fn += np.sum(((predict == 1) * (target == 2)) * (target > 0))
            np.save('tp.npy', tp)
            np.save('fp.npy', fp)
            np.save('fn.npy', fn)
            iou_list.append(iou)

            # if ix % 50 == 0:
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(np.array(vis_img[0,0].permute((1,2,0))))  # .transpose((1, 2, 0))
            #     plt.imshow(binary_mask[0, 0].cpu().detach().numpy(), alpha=0.5)
            #     # plt.imshow(gts[0, 0].cpu().detach().numpy(), alpha=0.5)
            #     # show_points(point[0].cpu(), point[1].cpu(), plt.gca())
            #     # show_points(points_small[0], input_label[0], plt.gca())
            #     plt.savefig('output_%s.png' %ix)
            if ix % 50 == 0:
                pbar.set_description(
                    '(Inference | {task}) Epoch {epoch} :: IoU {iou:.4f}'.format(
                        task='Eval',
                        epoch=0,
                        iou=np.nan_to_num(tp / (tp + fp + fn))))
        print('final IoU on val set is:', float(np.nan_to_num(tp / (tp + fp + fn))),flush=True)


        # Step 2: Format predictions and ground truth annotations into COCOeval format
        import json
        if 'kins' in dataset:
            coco_gt = COCO('ground_truth_annotations_kins_test.json')
            predictions_name = args['save_path']#predictions_kins.json'

        elif 'asd' in dataset:
            coco_gt = COCO('ground_truth_annotations.json')
            predictions_name = args['save_path']#'predictions_asd_wgt.json'
        elif 'kcar' in dataset:
            coco_gt = COCO('ground_truth_annotations_kcar_test.json')
            predictions_name = 'predictions_kcar_asdmodel.json'
        elif 'sailvos' in dataset:
            coco_gt = COCO('ground_truth_annotations_sailvos_val.json')
            predictions_name = 'predictions_sailvos_val.json'

        with open(predictions_name, 'w') as f:
            json.dump(predictions, f)

        coco_dt = coco_gt.loadRes(predictions_name)

        # Step 3: Use COCO API to evaluate predictions against ground truth annotations
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.params.useCats = 0
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Step 4: Calculate AP from the evaluation results
        # Assuming you want AP for a specific IoU threshold, e.g., 0.5
        average_precision = coco_eval.stats[0]  # AP at IoU threshold of 0.5
        print("Average Precision (AP) at IoU threshold 0.5:", average_precision)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-save_path', '--save_path', default='predictions_asd_wgt_saliency2.json', help='filename to save predictions', required=False)

    #adapt mode from: https://github.com/KidsWithTokens/Medical-SAM-Adapter/tree/main
    #normal mode is standard sam
    #parser.add_argument('-optimizer', type=str, default='adam', help='current options: adam, galore, aw')
    #parser.add_argument('-dataset', type=str, default='kins', help='current options: kins,asd, kcar')
    args = vars(parser.parse_args())
    main(args)