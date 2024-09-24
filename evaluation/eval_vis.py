# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
from collections import OrderedDict

#from mmcv.utils import print_log
import sys
sys.path.append('/beegfs/work/breitenstein/sam-pt/eval_asd')
from ytvis import YTVIS
from ytviseval import YTVISeval


def eval_vis(test_results, vis_anns, maxDets, logger=None):
    """Evaluation on VIS metrics.
    Args:
        test_results (dict(list[dict])): Testing results of the VIS dataset.
        vis_anns (dict(list[dict])): The annotation in the format
                of YouTube-VIS.
        logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
    Returns:
        dict[str, float]: Evaluation results.
    """
    ytvis = YTVIS(vis_anns)

    if len(ytvis.anns) == 0:
        #print_log('Annotations does not exist', logger=logger)
        print('Annotations does not exist')

        return

    ytvis_dets = ytvis.loadRes(test_results)
    vid_ids = ytvis.getVidIds()

    iou_type = metric = 'segm'
    eval_results = OrderedDict()
    ytvisEval = YTVISeval(ytvis, ytvis_dets, iou_type)
    ytvisEval.params.vidIds = vid_ids
    ytvisEval.params.useCats = 0
    ytvisEval.params.maxDets = maxDets

    ytvisEval.evaluate()
    ytvisEval.accumulate()

    # Save coco summarize print information to logger
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        ytvisEval.summarize()
    #print_log('\n' + redirect_string.getvalue(), logger=logger)
    print('\n' + redirect_string.getvalue())


    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'mAP50_p', 'mAP50_m', 'mAP50_h']
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,

        'mAP50_p': 6,
        'mAP50_m': 7,
        'mAP50_h': 8,

        'AR@1': 9,
        'AR@10': 10,
        'AR@100': 11,
        'AR_s@100': 12,
        'AR_m@100': 13,
        'AR_l@100': 14
    }

    for metric_item in metric_items:
        key = f'{metric}_{metric_item}'
        val = float(f'{ytvisEval.stats[coco_metric_names[metric_item]]:.5f}')
        eval_results[key] = val

    ap = ytvisEval.stats[:9]
    eval_results[f'{metric}_mAP_copypaste'] = (
        f'{ap[0]:.5f} {ap[1]:.5f} {ap[2]:.5f} {ap[3]:.5f} '
        # f'{ap[4]:.3f} {ap[5]:.3f}')
        f'{ap[4]:.5f} {ap[5]:.5f} {ap[6]:.5f} {ap[7]:.5f} {ap[8]:.5f}')
    return eval_results



# # Copyright (c) OpenMMLab. All rights reserved.
# import contextlib
# import io
# from collections import OrderedDict
#
# from mmcv.utils import print_log
#
# from .ytvis import YTVIS
# from .ytviseval import YTVISeval
#
#
# def eval_vis(test_results, vis_anns, maxDets, logger=None):
#     """Evaluation on VIS metrics.
#     Args:
#         test_results (dict(list[dict])): Testing results of the VIS dataset.
#         vis_anns (dict(list[dict])): The annotation in the format
#                 of YouTube-VIS.
#         logger (logging.Logger | str | None): Logger used for printing
#                 related information during evaluation. Default: None.
#     Returns:
#         dict[str, float]: Evaluation results.
#     """
#     ytvis = YTVIS(vis_anns)
#
#     if len(ytvis.anns) == 0:
#         print_log('Annotations does not exist', logger=logger)
#         return
#
#     ytvis_dets = ytvis.loadRes(test_results)
#     vid_ids = ytvis.getVidIds()
#
#     iou_type = metric = 'segm'
#     eval_results = OrderedDict()
#     ytvisEval = YTVISeval(ytvis, ytvis_dets, iou_type)
#     ytvisEval.params.vidIds = vid_ids
#
#     ytvisEval.params.maxDets = maxDets     #  maxDets  [1 10 100] M=3 thresholds on max detections per image
#
#     ytvisEval.evaluate()
#     ytvisEval.accumulate()
#
#     # Save coco summarize print information to logger
#     redirect_string = io.StringIO()
#     with contextlib.redirect_stdout(redirect_string):
#         ytvisEval.summarize()
#     print_log('\n' + redirect_string.getvalue(), logger=logger)
#
#     metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
#     coco_metric_names = {
#         'mAP': 0,
#         'mAP_50': 1,
#         'mAP_75': 2,
#         'mAP_s': 3,
#         'mAP_m': 4,
#         'mAP_l': 5,
#         'AR@1': 6,
#         'AR@10': 7,
#         'AR@100': 8,
#         'AR_s@100': 9,
#         'AR_m@100': 10,
#         'AR_l@100': 11
#     }
#
#     for metric_item in metric_items:
#         key = f'{metric}_{metric_item}'
#         val = float(f'{ytvisEval.stats[coco_metric_names[metric_item]]:.3f}')
#         eval_results[key] = val
#
#     ap = ytvisEval.stats[:6]
#     eval_results[f'{metric}_mAP_copypaste'] = (
#         f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
#         f'{ap[4]:.3f} {ap[5]:.3f}')
#     return eval_results