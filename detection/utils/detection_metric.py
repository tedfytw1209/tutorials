'''
Detection Metrics Function implement through MONAI. COCOMetrics
'''
from collections.abc import Sequence
from typing import Any
import numpy as np
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import box_utils

def mAP_with_IoU(
    pred_boxes: Sequence[np.ndarray],
    pred_classes: Sequence[np.ndarray],
    pred_scores: Sequence[np.ndarray],
    gt_boxes: Sequence[np.ndarray],
    gt_classes: Sequence[np.ndarray],
    classes: Sequence[str], 
    iou_range: Sequence[float] = (0.1, 0.5, 0.05),
    max_detection: Sequence[int] = (1, 5, 100),
    ):
    '''
    Args:
        pred_boxes: predicted boxes from single batch; List[[D, dim * 2]],
            D number of predictions
        pred_classes: predicted classes from a single batch; List[[D]],
            D number of predictions
        pred_scores: predicted score for each bounding box; List[[D]],
            D number of predictions
        gt_boxes: ground truth boxes; List[[G, dim * 2]], G number of ground
            truth
        gt_classes: ground truth classes; List[[G]], G number of ground truth
        classes (Sequence[str]): name of each class (index needs to correspond to predicted class indices!)
        iou_range (Sequence[float]): (start, stop, step) for mAP iou thresholds
        max_detection (Sequence[int]): maximum number of detections per image
    Return:
        mAP  and per class mAP result with IoU ranges (Dict[str, float])
    '''
    coco_metric = COCOMetric(classes=classes, iou_range=iou_range, iou_list=[0.1], max_detection=max_detection) #iou_list not need for mAP
    results_metric = matching_batch(
                    iou_fn=box_utils.box_iou,
                    iou_thresholds=coco_metric.iou_thresholds,
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    max_detections=max_detection[-1],
                )
    val_epoch_metric_dict = coco_metric(results_metric)[0]
    out_dic = {}
    for key in val_epoch_metric_dict.keys():
        if 'mAP_IoU' in key:
            out_dic[key] = val_epoch_metric_dict[key]
    
    return out_dic

def AP_at_IoU(
    pred_boxes: Sequence[np.ndarray],
    pred_classes: Sequence[np.ndarray],
    pred_scores: Sequence[np.ndarray],
    gt_boxes: Sequence[np.ndarray],
    gt_classes: Sequence[np.ndarray],
    classes: Sequence[str], 
    iou_list: Sequence[float] | float = 0.5,
    max_detection: Sequence[int] = (1, 5, 100),
    ):
    '''
    Args:
        pred_boxes: predicted boxes from single batch; List[[D, dim * 2]],
            D number of predictions
        pred_classes: predicted classes from a single batch; List[[D]],
            D number of predictions
        pred_scores: predicted score for each bounding box; List[[D]],
            D number of predictions
        gt_boxes: ground truth boxes; List[[G, dim * 2]], G number of ground
            truth
        gt_classes: ground truth classes; List[[G]], G number of ground truth
        classes (Sequence[str]): name of each class (index needs to correspond to predicted class indices!)
        iou_list (Sequence[float]): specific thresholds where ap is evaluated and saved
        max_detection (Sequence[int]): maximum number of detections per image
    Return:
        mAP  and per class mAP result with IoU ranges (Dict[str, float])
    '''
    if isinstance(iou_list,float):
        iou_list = [iou_list]
    coco_metric = COCOMetric(classes=classes, iou_list=iou_list) #iou_list not need for mAP
    results_metric = matching_batch(
                    iou_fn=box_utils.box_iou,
                    iou_thresholds=coco_metric.iou_thresholds,
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    max_detections=max_detection[-1],
                )
    val_epoch_metric_dict = coco_metric(results_metric)[0]
    out_dic = {}
    for key in val_epoch_metric_dict.keys():
        if 'AP_IoU' in key:
            for each_iou in iou_list:
                if f"AP_IoU_{each_iou:.2f}_" in key:
                    out_dic[key] = val_epoch_metric_dict[key]
    
    return out_dic
    
def mAR_with_IoU(
    pred_boxes: Sequence[np.ndarray],
    pred_classes: Sequence[np.ndarray],
    pred_scores: Sequence[np.ndarray],
    gt_boxes: Sequence[np.ndarray],
    gt_classes: Sequence[np.ndarray],
    classes: Sequence[str], 
    iou_range: Sequence[float] = (0.1, 0.5, 0.05),
    max_detection: Sequence[int] = (1, 5, 100),
    ):
    '''
    Args:
        pred_boxes: predicted boxes from single batch; List[[D, dim * 2]],
            D number of predictions
        pred_classes: predicted classes from a single batch; List[[D]],
            D number of predictions
        pred_scores: predicted score for each bounding box; List[[D]],
            D number of predictions
        gt_boxes: ground truth boxes; List[[G, dim * 2]], G number of ground
            truth
        gt_classes: ground truth classes; List[[G]], G number of ground truth
        classes (Sequence[str]): name of each class (index needs to correspond to predicted class indices!)
        iou_range (Sequence[float]): (start, stop, step) for mAP iou thresholds
        max_detection (Sequence[int]): maximum number of detections per image
    Return:
        mAP  and per class mAP result with IoU ranges (Dict[str, float])
    '''
    coco_metric = COCOMetric(classes=classes, iou_range=iou_range, iou_list=[0.1], max_detection=max_detection) #iou_list not need for mAP
    results_metric = matching_batch(
                    iou_fn=box_utils.box_iou,
                    iou_thresholds=coco_metric.iou_thresholds,
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    max_detections=max_detection[-1],
                )
    val_epoch_metric_dict = coco_metric(results_metric)[0]
    out_dic = {}
    for key in val_epoch_metric_dict.keys():
        if 'mAR_IoU' in key:
            out_dic[key] = val_epoch_metric_dict[key]
    
    return out_dic
