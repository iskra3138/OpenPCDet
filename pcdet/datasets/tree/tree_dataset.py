import copy
import pickle
import laspy

import numpy as np
from skimage import io

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils #, common_utils # object3d_kitti 
from ..dataset import DatasetTemplate

def get_objects_from_label(label_file): ## added for get_label
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

def cls_type_to_id(cls_type):
    type_to_id = {'tree': 1} # changed
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Object3d(object):  ## added for get_label
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1]) # 0: non-truncated, 1: truncated
        self.occlusion = float(label[2])  # the float number of 'ground points' / 'total number of points within a 5 m buffer'
        #self.alpha = 0.0 # set to zero, we don't need this one
        self.h = float(label[8])
        self.w = float(label[10])
        self.l = float(label[9])
        self.loc = np.array((float(label[11])-30, float(label[12])-30, float(label[13])-300), dtype=np.float32) # for flip augmentation along x,y-axis (only for 60*60 tiles)
        self.ry = 0.0 # set to zero, ## float(label[14])
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

class TreeDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.tree_infos = []  ## changed
        self.include_tree_data(self.mode) ## changed

    def include_tree_data(self, mode): ## changed
        if self.logger is not None:
            self.logger.info('Loading Tree dataset') ## changed
        tree_infos = [] ## changed

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                tree_infos.extend(infos) ## changed

        self.tree_infos.extend(tree_infos) ## changed

        if self.logger is not None:
            self.logger.info('Total samples for Tree dataset: %d' % (len(tree_infos))) ## changed

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None # list of file_names

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('T%s.las' % idx) ## chnaged
        assert lidar_file.exists()
        lasfile = laspy.file.File(lidar_file, mode="r") ## added
        return np.vstack((lasfile.x-30 , lasfile.y-30, lasfile.z-300)).transpose().astype(np.float32) # must be same with line 37!!!!!

    def get_label(self, idx): # everything is modified
        label_file = self.root_split_path / 'labels' / ('%s.txt' % idx) 
        assert label_file.exists()

        return get_objects_from_label(label_file)

    # get_infos is used for making pkl files
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None): ## modified
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 3, 'lidar_idx': sample_idx} ## changed
            info['point_cloud'] = pc_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                #annotations['alpha'] = np.array([obj.alpha for obj in obj_list]) # all is 0
                #annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.w, obj.h] for obj in obj_list]) # lwh(lidar) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                #num_objects = len([obj.cls_type for obj in obj_list]) #if obj.cls_type != 'DontCare'])
                #num_gt = len(annotations['name'])
                num_objects = len(annotations['name'])
                index = list(range(num_objects)) #+ [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location']  #[:num_objects]
                dims = annotations['dimensions'] #[:num_objects]
                rots = annotations['rotation_y'] #[:num_objects]
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3] # Changed!!!! for lwh format
                gt_boxes_lidar = np.concatenate([loc, l, w, h, rots[..., np.newaxis]], axis=1) # changed
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_objects, dtype=np.int32) #(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(points, corners_lidar[k]) ## changed
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'): ## modified
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('tree_dbinfos_%s.pkl' % split)  ## changed

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            #bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    #gt_points.astype(np.float32).tofile(f)
                    gt_points.tofile(f)  ## gt_point should be float32 and this file is used for sampling augmentation

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,  ## changed
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'score': annos['score'][i]} ## changed
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                #'bbox': np.zeros([num_samples, 4]), 
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            #pred_dict['alpha'] = np.zeros_like(pred_boxes[:, 6]) #-np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes[:, 6]
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = np.zeros_like(pred_boxes[:, 6]) # set to 0.0, original code was pred_boxes[:, 6]
            #pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['boxes_lidar'][:, 6] = 0.0 # changed

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    #bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lwh
                    for idx in range(len(dims)):
                        priint ('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], #single_pred_dict['alpha'][idx],
                                 #bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.tree_infos[0].keys(): ## changed
            return None, {}

        from .tree_object_eval_python import eval_tree as tree_eval ## changed

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.tree_infos] ## changed

        ap_result_str, ap_dict = tree_eval.get_tree_eval_result(eval_gt_annos, eval_det_annos, class_names) ## changed
        
        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.tree_infos) * self.total_epochs ## changed

        return len(self.tree_infos)  ## changed

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.tree_infos) ## changed

        info = copy.deepcopy(self.tree_infos[index])  ## changed

        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            annos = info['annos']
            #annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar'] # added

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_tree_infos(dataset_cfg, class_names, data_path, save_path, workers=4): ## modified
    print (80*'#')
    dataset = TreeDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('tree_infos_%s.pkl' % train_split) ## changed
    val_filename = save_path / ('tree_infos_%s.pkl' % val_split) ## changed
    trainval_filename = save_path / 'tree_infos_trainval.pkl' ## changed
    test_filename = save_path / 'tree_infos_test.pkl' ## changed

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    tree_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True) ## changed
    with open(train_filename, 'wb') as f:
        pickle.dump(tree_infos_train, f) ## changed
    print('Tree info train file is saved to %s' % train_filename) ## changed

    dataset.set_split(val_split)
    tree_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True) ## changed
    with open(val_filename, 'wb') as f:
        pickle.dump(tree_infos_val, f) ## changed
    print('Tree info val file is saved to %s' % val_filename) ## changed

    with open(trainval_filename, 'wb') as f:
        pickle.dump(tree_infos_train + tree_infos_val, f) ## changed
    print('Tree info trainval file is saved to %s' % trainval_filename) ## changed

    dataset.set_split('test')
    tree_infos_test = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True) ## changed for getting results for testset
    with open(test_filename, 'wb') as f:
        pickle.dump(tree_infos_test, f) ## changed
    print('Tree info test file is saved to %s' % test_filename) ## changed

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_tree_infos': ## changed
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_tree_infos( ## changed
            dataset_cfg=dataset_cfg,
            class_names=['tree'],
            data_path=ROOT_DIR / 'data' / 'tree',
            save_path=ROOT_DIR / 'data' / 'tree'
        )
