# translates the results to video results
# performs point tracking for full occlusions
# 2 stage evaluation more efficient than everything together
# 1 stage evaluation possible and results do not change
# our own code for S-AModal


import cvbase as cvb
import numpy as np
import os
from pycocotools import mask
import json
from pycocotools.coco import COCO
from ..evaluation.ap_evaluation import customCOCOeval as COCOeval
from ..evaluation.eval_vis import eval_vis
from sam_pt.point_tracker.cotracker import CoTrackerPointTracker
import torch
import cv2
import matplotlib.pyplot as plt
from ..utils.helper_functions import *





pred_data_sam = '/beegfs/work/breitenstein/segment-anything/predictions_asd_wgt_run1.json'
coco_preds = cvb.load(pred_data_sam)
imageinfo = cvb.load('/beegfs/work/breitenstein/segment-anything/imgage_info_asd_val.json')
#imgs_dict = cvb.load('/beegfs/work/breitenstein/segment-anything/imgs_dict_asd_val.json')
ytvis_format_sampreds = []
unique_tracking_ids = []
videoimagepath = '/beegfs/work/breitenstein/sam-pt/data/amodal_synth_drive/JPEGImages/'
videos = [f for f in sorted(os.listdir(videoimagepath)) if f.endswith('D')]

height = 1080
width = 1920

tracking_dict = {}
for ann in coco_preds:
    ann['track_id'] = int(str(ann['track_id'])+str(int(ann['image_id']/100))) #fix for multiple tracking ids
    if ann['track_id'] in unique_tracking_ids:
        continue
    else:
        unique_tracking_ids.append(ann['track_id'])
        tracking_dict[ann['track_id']] = {}
        tracking_dict[ann['track_id']]['color'] = [255, 255, 0]
        # set color to yellow to visualize amodal masks for visibility reasons
        # [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]

counter = 0

# the following converts the image-level predictions to video-level predictions
for track_id in unique_tracking_ids:

    predictions_per_pred = [ann for ann in coco_preds if ann['track_id']==track_id]
    example_ann = predictions_per_pred[0]
    annotation_dict = {'segmentations': [], 'points': [], 'areas': [], 'id': counter, 'track_id': track_id}
    ann_image_name = [ann for ann in imageinfo if imageinfo[ann]['image_id'] == example_ann['image_id']]
    ann_image_info = imageinfo[ann_image_name[0]]
    # ann_image_info has field image_id (in 0,100) and 'video'
    annotation_dict['video_id'] = videos.index(ann_image_info['video'])
    annotation_dict['category_id'] = example_ann['category_id']
    annotation_dict['scores'] = []
    annotation_dict['eval'] = []
    annotation_dict['score'] = None

    unique_image_ids = []
    video_multiplier = annotation_dict['video_id']*100
    frame_counter = 0
    empty_dict = {'size': [1080,1920], 'counts': 'PPYo1'}
    for ann in predictions_per_pred:
        # we need the precise image id to be able to add none as is needed
        ann_image_name1 = [ann1 for ann1 in imageinfo if imageinfo[ann1]['image_id'] == ann['image_id']]
        ann_image_info1 = imageinfo[ann_image_name1[0]]
        if video_multiplier>0:
            image_id = ann_image_info1['image_id'] % video_multiplier
        else:
            image_id = ann_image_info1['image_id']
        if image_id==frame_counter:
            annotation_dict['segmentations'].append(ann['segmentation'])
            annotation_dict['scores'].append(ann['score'])
            annotation_dict['points'].append(ann['original_point'])
            annotation_dict['areas'].append(int(mask.area(ann['segmentation'])))
            frame_counter += 1
        elif image_id>frame_counter:
            while frame_counter<image_id:
                annotation_dict['segmentations'].append(empty_dict)
                annotation_dict['points'].append(None)
                annotation_dict['areas'].append(None)
                #for now do not add 'areas' or 'scores'
                #for 'score' only final mean value is interesting
                frame_counter += 1
            annotation_dict['segmentations'].append(ann['segmentation'])
            annotation_dict['scores'].append(ann['score'])
            annotation_dict['areas'].append(int(mask.area(ann['segmentation'])))
            annotation_dict['points'].append(ann['original_point'])
            frame_counter += 1
    while frame_counter < 100: #this apparently influences the result? why?
        annotation_dict['segmentations'].append(empty_dict)
        annotation_dict['points'].append(None)
        annotation_dict['areas'].append(None)
        frame_counter += 1

    annotation_dict['score']= np.mean(annotation_dict['scores'])
    ytvis_format_sampreds.append(annotation_dict)
    counter += 1

# save video level predictions
with open('asd_predictions_sam_converted_erosion.json', 'w') as f:
    json.dump(convert_to_json_serializable(ytvis_format_sampreds), f)


print('\n\n------------ASD amodal video-level evaluation------------\n')
coco_gt = COCO('/beegfs/work/breitenstein/segment-anything/ground_truth_annotations.json')
amodal_gt_asd_video = cvb.load('/beegfs/work/breitenstein/sam-pt/scripts/video_gt_asd_val.json')
eval_results = dict()
res_total = {}
amodal_vis_eval_results = eval_vis(
    test_results='asd_predictions_sam_converted_erosion.json',
    vis_anns=amodal_gt_asd_video,
    maxDets=[1, 10, 100])
eval_results.update(amodal_vis_eval_results)
res_total["amodal per video segm mAP"] = {}
res_total["amodal per video segm mAP"] = amodal_vis_eval_results


for key_res in res_total.keys():
    print("\n{} is {}\n".format(key_res, res_total[key_res]))

print('\n\n------------ASD amodal image-level evaluation------------\n')

pred_coco = coco_gt.loadRes(pred_data_sam)  # Use loadRes to load predictions

coco_eval = COCOeval(coco_gt, pred_coco, iouType='segm')
coco_eval.params.useCats = 0
# coco_eval.params.imgIds = coco_eval.params.imgIds[0:100]
# Perform evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


#  point tracking for full occlusions:
interp_shape = [384, 512]
visibility_threshold = 0.7
support_grid_size = 2
support_grid_every_n_frames = 12
add_debug_visualisations = False
pointpath = '/beegfs/work/breitenstein/sam-pt/models/cotracker_ckpts/cotracker_stride_4_wind_8.pth'
point_tracker = CoTrackerPointTracker(pointpath, interp_shape, visibility_threshold,
                                      support_grid_size, support_grid_every_n_frames, add_debug_visualisations)


# set option for recovery:
all_images = True
update_points = False

if all_images:
    # path to save visualizations
    savepath = '/beegfs/work/breitenstein/sam-pt/eval_asd/asd_vis_check/'

point_tracker.to(device='cuda')

# now we iterate through the predicitons:
ytvis_format_sampreds_wocclusion = []

for ann in ytvis_format_sampreds:
    not_visible_yet=True
    track_id = ann['track_id']
    # if track_id != 1034280:
    #     continue
    video_id = ann['video_id']
    video = videos[video_id]
    overlay_color = tracking_dict[track_id]['color']  # Red: [R, G, B]
    alpha = 0.5  # Adjust as needed, 0 is fully transparent, 1 is fully opaque

    if not os.path.exists(savepath + video):
        os.mkdir(savepath + video)
    if not os.path.exists(savepath + video + '/' + str(track_id)):
        os.mkdir(savepath + video + '/' + str(track_id))
    frames = sorted(os.listdir(videoimagepath + video))
    cur_occluded = False
    ann_new = ann.copy()  # copy to modify for with occlusion treatment
    ann_new['displacement'] = []  # add a field for saving the displacement
    point_list = []
    for k in range(0,len(ann['segmentations'])):  # go through all segmentations

        if ann['segmentations'][k]['counts']=='PPYo1' and not_visible_yet:
            continue  # if empty mask and not visible yet, we can just continue
        elif ann['segmentations'][k]['counts'] != 'PPYo1':
            point_list.append(ann['points'][k])
            not_visible_yet = False
            last_appearance = k
            cur_occluded = False
            # Define the transparency level (alpha) for the overlay
            # Create a copy of the original image to overlay
            image2 = cv2.imread(videoimagepath + video + '/' + frames[k])
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            overlay_image = np.copy(image2).astype('uint8')

            # Create a mask for the overlay based on the binary mask
            overlay_mask = np.zeros_like(overlay_image)
            sam_mask = mask.decode(ann['segmentations'][k])
            overlay_mask[
                sam_mask == 1] = overlay_color  # Set alpha channel to 0 for fully transparent areas
            # Combine the original image and the overlay with transparency
            overlay_image = (1 - alpha) * overlay_image + alpha * overlay_mask
            plt.imsave(savepath + video + '/' + str(track_id) + '/' + '%s_%s.png' % (frames[k].split('_rgb')[0], track_id),
                       overlay_image.astype('uint8'))
            plt.imsave(savepath + video + '/' + str(track_id) + '/' + 'mask_%s_%s.png' % (frames[k].split('_rgb')[0], track_id),
                       sam_mask.astype('uint8'))
            continue
        elif ann['segmentations'][k]['counts'] == 'PPYo1' and not not_visible_yet:
            # this is the interesting case of full occlusion
            # we need the current frame
            image_id = k
            if not cur_occluded:
                cur_occluded = True
                print('instance is now fully occluded in frame k', k, flush=True)
                occlusion_start = image_id - 1

                last_appearance = ann['points'][image_id-1]
                # need this to get rid of instances that are vanishing
                last_appearance_sanity_check = ann['points'][image_id-1]

                print('last appearance was ', last_appearance, flush=True)
                print('occlusion_start was ', occlusion_start, flush=True)
                if not last_appearance:
                    last_appearance = ann['points'][image_id-1]
            else:
                cur_occluded = True

            # get rid of an instance whose last amodal mask was already close to the boundary
            # get last predicted mask:
            last_mask_bbox = mask.toBbox(ann['segmentations'][k-1])
            x_coords = [last_mask_bbox[0],last_mask_bbox[0]+last_mask_bbox[2]]
            y_coords = [last_mask_bbox[1], last_mask_bbox[1] + last_mask_bbox[3]]
            if x_coords[1] >= width -20:
                vanished =True
                continue
            elif x_coords[0] <= 20:
                vanished = True
                continue
            if last_appearance_sanity_check[0] >= width-20:
                vanished = True
                continue
            elif last_appearance_sanity_check[0] <= 20:
                vanished = True
                continue

            if not update_points:
                image2 = cv2.imread(videoimagepath + video + '/' + frames[occlusion_start])
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                rgbs = torch.tensor(image2[None])

                for j in range(occlusion_start+1, image_id + 1):
                    image2 = cv2.imread(videoimagepath + video + '/' + frames[j])
                    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                    image2 = torch.tensor(image2)
                    rgbs = torch.concat([rgbs, image2[None]], dim=0)
            else:
                image2 = cv2.imread(videoimagepath + video + '/' + frames[0])
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                query_point = torch.tensor([0, ann['points'][0][0], ann['points'][0][1]])[None]
                rgbs = torch.tensor(image2[None])

                for j in range(1, image_id + 1):
                    image2 = cv2.imread(videoimagepath + video + '/' + frames[j])
                    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                    image2 = torch.tensor(image2)
                    rgbs = torch.concat([rgbs, image2[None]], dim=0)
                    if j <= occlusion_start:
                        query_point = torch.cat([query_point,torch.tensor([j, ann['points'][j][0], ann['points'][j][1]])[None]])

            # calculate the average displacement in points list:
            avg_displacement_x = []
            avg_displacement_y = []
            # sum_x = 0
            # sum_y = 0
            for p1 in range(0,len(point_list)-1):
                avg_displacement_x.append(abs(point_list[p1][0] - point_list[p1+1][0]))
                avg_displacement_y.append(abs(point_list[p1][1] - point_list[p1 + 1][1]))
                # sum_x += (point_list[p1][0] - point_list[p1+1][0])
                # sum_y += (point_list[p1][1] - point_list[p1 + 1][1])

            rgbs = rgbs.permute((0, 3, 1, 2))
            # previous_mask_idx = last_appearance

            query_point = torch.tensor([0,last_appearance[0], last_appearance[1]])

            with torch.no_grad():
                pred_point, visibility = point_tracker(rgbs[None].to('cuda'), query_point[None,None].to('cuda'))

            # displacement vector from last point to before

            d1 = (pred_point[0, -1, 0, 0] - last_appearance[0]).cpu().numpy()
            d2 = (pred_point[0, -1, 0, 1] - last_appearance[1]).cpu().numpy()
            # check if displacement same as average displacement:
            if abs(d1) > np.mean(avg_displacement_x)*(rgbs.shape[0]-1):
                d1 = np.mean(avg_displacement_x)*(rgbs.shape[0]-1)*np.sign(d1)
            if abs(d2) > np.mean(avg_displacement_y)*(rgbs.shape[0]-1):
                d2 = np.mean(avg_displacement_y)*(rgbs.shape[0]-1)*np.sign(d2)

            displacement = (d1,d2)  # last_appearance[0][1] if only 1 point

            previous_mask = mask.decode(ann_new['segmentations'][occlusion_start])
            ann_new['points'][image_id] = [pred_point[0, -1, 0, 0], pred_point[0, -1, 0, 1]]
            # Move the previous mask to the current frame
            shifted_mask = move_mask(previous_mask, displacement)
            shifted_mask = shifted_mask.astype(np.uint8)
            shifted_mask = np.asfortranarray(shifted_mask)
            shifted_mask_encoded = mask.encode(shifted_mask)
            shifted_mask_encoded['counts'] = str(shifted_mask_encoded['counts'], 'utf-8')
            area = mask.area(shifted_mask_encoded)

            ann_new['segmentations'][image_id] = shifted_mask_encoded
            ann_new['areas'][image_id] = int(area)
            # Define the transparency level (alpha) for the overlay
            # Create a copy of the original image to overlay
            overlay_image = np.copy(image2).astype('uint8')
            # Create a mask for the overlay based on the binary mask
            overlay_mask = np.zeros_like(overlay_image)
            overlay_mask[
                shifted_mask == 1] = overlay_color  # Set alpha channel to 0 for fully transparent areas
            # Combine the original image and the overlay with transparency
            overlay_image = (1 - alpha) * overlay_image + alpha * overlay_mask
            plt.imsave(savepath + video + '/' + str(track_id) + '/' + '%s_%s.png' % (frames[image_id].split('_rgb')[0], track_id),
                       overlay_image.astype('uint8'))
            plt.imsave(savepath + video + '/' + str(track_id) + '/' + 'mask_%s_%s.png' % (frames[k].split('_rgb')[0], track_id),
                       shifted_mask.astype('uint8'))

    ytvis_format_sampreds_wocclusion.append(ann_new)

# save json with full occlusions
with open('asd_predictions_sam_converted_occlusions.json', 'w') as f:
    json.dump(convert_to_json_serializable(ytvis_format_sampreds_wocclusion), f)

print('\n\n------------ASD amodal video-level evaluation------------\n')
eval_results = dict()
res_total = {}
amodal_vis_eval_results = eval_vis(
    test_results='asd_predictions_sam_converted_occlusions.json',
    vis_anns=amodal_gt_asd_video,
    maxDets=[1, 10, 100])
eval_results.update(amodal_vis_eval_results)
res_total["amodal per video segm mAP"] = {}
res_total["amodal per video segm mAP"] = amodal_vis_eval_results


for key_res in res_total.keys():
    print("\n{} is {}\n".format(key_res, res_total[key_res]))