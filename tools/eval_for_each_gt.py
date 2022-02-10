import argparse
import glob
from pathlib import Path
import laspy

import numpy as np
import torch
import os
import datetime

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_cuda

def cls_type_to_id(cls_type):
    type_to_id = {'tree': 1}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Object3d(object):  ## added for get_label
    def __init__(self, line):
    #def __init__(self, line, min_z):    
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1]) # 0: non-truncated, 1: truncated
        self.occlusion = float(label[2])  # the float number of 'ground points' / 'total number of points within a 5 m buffer'
        self.alpha = 0.0 # we don't need this one
        #self.box2d = np.array((0.0, 0.0, 50.0, 50.0), dtype=np.float32) # we don't need this one
        self.h = float(label[8])
        self.w = float(label[10])
        self.l = float(label[9])
        self.box2d = np.array((float(label[11]) - self.l/2 ,
                               float(label[12]) - self.w/2,
                               float(label[11]) + self.l/2,
                               float(label[12]) + self.w/2), dtype=np.float32)
        self.loc = np.array((float(label[11])-30, float(label[12])-30, float(label[13])-300), dtype=np.float32)
        #self.loc = np.array((float(label[11]), float(label[12]), float(label[13])-min_z), dtype=np.float32)
        #self.dis_to_cam = np.linalg.norm(self.loc) # we don't need this one
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_tree_obj_level()

    def get_tree_obj_level(self):
        if self.occlusion > 0.8:
            self.level_str = 'Easy'
            return 0  # Easy
        elif self.occlusion > 0.2:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif self.occlusion <= 0.2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

class EvalDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path='../data/tree', txt_file=None, path=None, ext='.las', logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            root_path:
            txt_file:
            path:
            ext:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.ext = ext
        #data_file_list = glob.glob(str(file_path / f'*{self.ext}')) if file_path.is_dir() else [file_path]
        with open(txt_file, 'r') as f :
            lines = f.readlines()
        data_file_list = []
        label_file_list = []
        for line in lines :
            data_file = os.path.join(root_path, path, 'velodyne', 'T{}.las'.format(line.strip()))
            assert os.path.exists(data_file)
            data_file_list.append(data_file)
            label_file = os.path.join(root_path, path, 'labels', '{}.txt'.format(line.strip()))
            assert os.path.exists(label_file)
            label_file_list.append(label_file)
                    
        #data_file_list.sort()
        self.sample_file_list = data_file_list

        self.label_file_list = label_file_list
        
    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.las':
            lasfile = laspy.file.File(self.sample_file_list[index], mode="r")
            points = np.vstack((lasfile.x-30 , lasfile.y-30
                                 , lasfile.z-300, np.zeros_like(lasfile.x) )).transpose()
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        with open(self.label_file_list[index], 'r') as f:
            lines = f.readlines()
        obj_list = [Object3d(line) for line in lines]
        annotations = {}
        annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
        annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
        annotations['alpha'] = np.array([obj.alpha for obj in obj_list]) # all is 0
        annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
        annotations['score'] = np.array([obj.score for obj in obj_list])
        annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
        num_gt = len(annotations['name'])
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)

        loc = annotations['location']
        dims = annotations['dimensions']
        rots = annotations['rotation_y']
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        gt_boxes_lidar = np.concatenate([loc, l, w, h, rots[..., np.newaxis]], axis=1) # changed
        #annotations['gt_boxes_lidar'] = gt_boxes_lidar
        

        input_dict = {
            'points': points[:,:3],
            'frame_id': index,
            'gt_boxes': gt_boxes_lidar,
            'truncated': annotations['truncated'],
            'occluded' : annotations['occluded'],
            'gt_names': annotations['name']
        }
        #print ('Points shape before_prepare_data: ', input_dict['points'].shape)
        data_dict = self.prepare_data(data_dict=input_dict)
        #print ('Points shape after_prepare_data: ', data_dict['points'].shape)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/tree_models/pointpillar.yaml',
                        help='specify the config for evaluation')
    parser.add_argument('--txt_file', type=str, default='./../data/tree/ImageSets/val.txt',
                        help='specify txt file that contains file name for evaluation')
    parser.add_argument('--path', type=str, default='training', 
                        help="specify the path['training' or 'testing'] that have files for evaluation") 
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.las', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    #boxes_a = torch.tensor(boxes_a.astype(np.float32)).cuda()
    #boxes_b = torch.tensor(boxes_b.astype(np.float32)).cuda()

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


def main():
    args, cfg = parse_config()
    time_info = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    txt_name = os.path.split(args.txt_file)[1][:-4]
    log_path = './log_of_eval_for_each_gt'
    if not os.path.exists(log_path) :
        os.makedirs(log_path)
    log_file = os.path.join(log_path, 'log_{}_{}.txt'.format(txt_name, time_info))
    logger = common_utils.create_logger(log_file)
    logger.info('-----------------Evaluation of Each Files-------------------------')
    eval_dataset = EvalDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path('./../data/tree'),txt_file = args.txt_file, path = args.path, 
        ext=args.ext, logger=logger
    )

    logger.info(f'Total number of samples: \t{len(eval_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=eval_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    f = open(os.path.join(log_path,'result_{}_{}.csv'.format(txt_name, time_info)), 'w')
    head_line = 'file_id, tree_id, results, IoU, score, truncated, occluded, GT_X, GT_Y, GT_Z, GT_l, GT_w, GT_h, GT_r, PD_X, PD_Y, PD_Z, PD_l, PD_w, PD_h, PD_r\n'
    f.write(head_line)

    with torch.no_grad():
        for idx, data_dict in enumerate(eval_dataset):
            file_name = os.path.split(eval_dataset.label_file_list[idx])[1][:-4]
            logger.info (100*'#')
            logger.info(f'Visualized sample index: {file_name}')

            data_dict = eval_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            gt_list = [x for x in range(data_dict['gt_boxes'][0][:,:7].shape[0])]
            pd_list = [x for x in range(pred_dicts[0]['pred_boxes'].shape[0])]

            logger.info ('Prediction Results')
            preds_log = ''
            for i in pd_list :
                preds_log += 'score: {:>10.4f} ,'.format(pred_dicts[0]['pred_scores'][i])
                preds_log += 'box : ['
                for j in range(7):
                    preds_log += '{:>10.4f} '.format(pred_dicts[0]['pred_boxes'][i][j])
                preds_log += ']\n'
            logger.info (f"preds: \n{preds_log}")

            gt_log = ''
            for i in gt_list :
                gt_log += '['
                for j in range(7) :
                    gt_log += '{:>10.4f} '.format(data_dict['gt_boxes'][0][i,j])
                gt_log += ']\n'
            logger.info(f"GT \n{gt_log}")

            if len(pd_list) != 0 :
                pred_dicts[0]['pred_boxes'][:,-1] = 0.0
                iou = boxes_iou3d_gpu(data_dict['gt_boxes'][0][:,:7], pred_dicts[0]['pred_boxes'])
                assert len(gt_list)==iou.shape[0]
                assert len(pd_list)==iou.shape[1]
                iou_log = ''
                for i in range(iou.shape[0]):
                    iou_log += '['
                    for j in range(iou.shape[1]):
                        iou_log += '{:>10.4f}'.format(iou[i][j])
                    iou_log += ']\n'
                logger.info(f"IoU \n{iou_log}")

                while np.max(iou) > 0.1 :
                    r, c = np.unravel_index(np.argmax(iou, axis=None), iou.shape)
                    result_str = '{}, {}, detected, {:.2f}, {:.2f}, '.format(file_name, r, np.max(iou), pred_dicts[0]['pred_scores'][c])
                    result_str += '{}, {}, '.format(data_dict['truncated'][0][r], data_dict['occluded'][0][r])
                    for g in data_dict['gt_boxes'][0][r,:7]:
                        result_str += '{:.2f}, '.format(g)
                    for d in pred_dicts[0]['pred_boxes'][c][:-1] :
                        result_str += '{:.2f}, '.format(d)
                    result_str += '{:.2f}\n'.format(pred_dicts[0]['pred_boxes'][c][-1])
                    f.write(result_str)
                    iou[r,:] = -1.0
                    iou[:,c] = -1.0
                    gt_list.remove(r)
                    pd_list.remove(c)
                    
                # missed from gt_list
                for r in gt_list :
                    result_str = '{}, {}, missed, , , '.format(file_name, r)
                    result_str += '{}, {}, '.format(data_dict['truncated'][0][r], data_dict['occluded'][0][r])
                    for g in data_dict['gt_boxes'][0][r,:7]:
                        result_str += '{:.2f}, '.format(g)
                    
                    result_str += ', , , , , , \n'
                    f.write(result_str)

                # mispredict from pd_list
                for c in pd_list :
                    result_str = '{}, , mispredict, , {:.2f}, '.format(file_name, pred_dicts[0]['pred_scores'][c])
                    result_str += ', , '
                    result_str += ', , , , , , , '
                    for d in pred_dicts[0]['pred_boxes'][c][:-1] :
                        result_str += '{:.2f}, '.format(d)
                    result_str += '{:.2f}\n'.format(pred_dicts[0]['pred_boxes'][c][-1])
                    f.write(result_str)
            else :
                # missed from gt_list
                for r in gt_list :
                    result_str = '{}, {}, missed, , , '.format(file_name, r)
                    result_str += '{}, {}, '.format(data_dict['truncated'][0][r], data_dict['occluded'][0][r])
                    for g in data_dict['gt_boxes'][0][r,:7]:
                        result_str += '{:.2f}, '.format(g)

                    result_str += ', , , , , , \n'
                    f.write(result_str)

    logger.info('Evaluation done.')
    f.close()

if __name__ == '__main__':
    main()
