import argparse
import glob
from pathlib import Path
import laspy

import numpy as np
import torch
import os

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

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

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path='../data/tree', file_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.ext = ext
        data_file_list = glob.glob(str(file_path / f'*{self.ext}')) if file_path.is_dir() else [file_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

        self.labelfile = os.path.join('../data/tree/training/labels/' , os.path.split(file_path)[1][1:-4]+'.txt')
        
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

        with open(self.labelfile, 'r') as f:
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
            'gt_names': annotations['name']
        }
        print ('Points shape before_prepare_data: ', input_dict['points'].shape)
        data_dict = self.prepare_data(data_dict=input_dict)
        print ('Points shape after_prepare_data: ', data_dict['points'].shape)

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.las', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path('./../data/tree'),file_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            #logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            #print (data_dict['points'][:,1:])
            pts = data_dict['points'][:,1:]
            print ("POINT CLOUD RANGE: [{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}]".format(
                pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max(), pts[:,2].min(), pts[:,2].max()))

            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            print (100 * '#')
            print ('Prediction Results')
            for pred_dict in pred_dicts :
                for k, v in pred_dict.items() :
                    print (k, v.cpu())

            print (100 * '#')
            print ('GT BBoes')
            print (data_dict['gt_boxes'][0][:,:7].cpu())
            #V.draw_scenes(
            #    points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0], ref_boxes=pred_dicts[0]['pred_boxes'],
            #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            #)

            #if not OPEN3D_FLAG:
            #    mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
