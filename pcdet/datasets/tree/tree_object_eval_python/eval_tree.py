import io as sysio

import numba
import numpy as np
import torch

from .rotate_iou import rotate_iou_gpu_eval
from ....ops.iou3d_nms import iou3d_nms_cuda

@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


#@numba.jit(nopython=True, parallel=True)
@numba.jit(nopython=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    boxes_a = torch.tensor(boxes_a.astype(np.float32)).cuda()
    boxes_b = torch.tensor(boxes_b.astype(np.float32)).cuda()

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d.cpu().numpy()


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis.
    Args:
        gt_annos: dict
        dt_annos: dict
        metric: eval type. 0: bbox, 1: 3d # For tree, bev is smae to bbox.
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            #overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
            overlap_part = boxes_iou3d_gpu(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part
    
    return overlaps, total_gt_num, total_dt_num



def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def eval_class(gt_annos, dt_annos, metric, iou_th):
    metric_names = ['BEV', '3d']
    case_name = '{}_{}'.format(metric_names[metric], iou_th)

    overlaps, total_gt_num, total_dt_num = calculate_iou_partly(
            gt_annos, dt_annos, metric, num_parts=100)

    confidence = []
    tp = []
    fp = []
    dificulties = []
    for i, overlap in enumerate(overlaps) :
        num_gt_per_file, num_dt_per_file = overlap.shape
        tp_per_file = np.zeros(num_dt_per_file)
        fp_per_file = np.ones(num_dt_per_file)
        dificulties_per_file = np.zeros(num_dt_per_file) - 1.
        #print (i, num_gt_per_file, num_dt_per_file)

        gt_dificulties = gt_annos[i]['difficulty']
        gt_indices, dt_indices = np.where(overlap > iou_th)
        num_tp_per_file = len(dt_indices) 

        if num_tp_per_file == 0 :
            continue
        else :
            tp_per_file[dt_indices] = 1
            fp_per_file[dt_indices] = 0
            dificulties_per_file[dt_indices] = gt_dificulties[gt_indices]

        #print (gt_annos[i]['difficulty'])

        conf_per_file = dt_annos[i]['score']

        confidence.append(conf_per_file)
        tp.append(tp_per_file)
        fp.append(fp_per_file)
        dificulties.append(dificulties_per_file)

    confidence_fn = np.concatenate(confidence, axis=0)
    tp_fn = np.concatenate(tp, axis=0)
    fp_fn = np.concatenate(fp, axis=0)
    dfct_fn = np.concatenate(dificulties, axis=0)

    sorted_ind = np.argsort(-confidence_fn)
    sorted_conf = sorted(confidence_fn,reverse=True)

    sorted_tp = tp_fn[sorted_ind]
    sorted_fp = fp_fn[sorted_ind]
    sorted_tp = np.cumsum(sorted_tp)
    sorted_fp = np.cumsum(sorted_fp)

    sorted_dfct = dfct_fn[sorted_ind]

    rec = sorted_tp / float(total_gt_num.sum())
    prec = sorted_tp / np.maximum((sorted_tp + sorted_fp), np.finfo(np.float32).eps)

    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    result = {}
    result['experiment'] = case_name
    result['metric'] = metric
    result['iou_th'] = iou_th
    result['total number of gt'] = total_gt_num.sum()
    result['total number of prediction'] = total_dt_num.sum()
    result['scores'] = sorted_conf
    result['tps'] = sorted_tp
    result['fps'] = sorted_fp
    result['dificulties'] = sorted_dfct
    result['recalls'] = rec
    result['precisions'] = prec
    result['precision'] = prec[-1]
    result['recall'] = rec[-1]
    result['ap'] = ap
    
    return result


def get_tree_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):

    min_overlaps = [0.3, 0.5, 0.7]

    ret_2d = [] # For 2d BEV results
    ret_3d = [] # For 3d results
    for min_overlap in min_overlaps :
        iou_th = min_overlap
        ret_2d.append(eval_class(gt_annos, dt_annos, 0, iou_th))
        ret_3d.append(eval_class(gt_annos, dt_annos, 1, iou_th))

    result = '\n'
    for i in range(len(min_overlaps)):
        result += print_str(
            (f"{'tree'} "
             "Results@{:.2f}:".format(min_overlaps[i])))
        result += print_str((f"BEV AP: {ret_2d[i]['ap']:.4f}, "
                             f"BEV precision: {ret_2d[i]['precision']:.4f} "
                             f"BEV recall: {ret_2d[i]['recall']:.4f} "))
        result += print_str((f"3d AP: {ret_3d[i]['ap']:.4f}, "
                             f"3d precision: {ret_3d[i]['precision']:.4f} "
                             f"3d recall: {ret_3d[i]['recall']:.4f} "
                            ))

    ret_dict = {}
    for i in range(len(min_overlaps)):
        ret_dict['{}_BEV/iou_{}_AP'.format('tree', min_overlaps[i])] = ret_2d[i]['ap']
        ret_dict['{}_BEV/iou_{}_precision'.format('tree', min_overlaps[i])] = ret_2d[i]['precision']
        ret_dict['{}_BEV/iou_{}_recall'.format('tree', min_overlaps[i])] = ret_2d[i]['recall']

        ret_dict['{}_3D/iou_{}_AP'.format('tree', min_overlaps[i])] = ret_3d[i]['ap']
        ret_dict['{}_3D/iou_{}_precision'.format('tree', min_overlaps[i])] = ret_3d[i]['precision']
        ret_dict['{}_3D/iou_{}_recall'.format('tree', min_overlaps[i])] = ret_3d[i]['recall']

    return result, ret_2d, ret_3d, ret_dict
