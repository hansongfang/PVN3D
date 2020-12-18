from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import torch
from loguru import logger
from torch.utils.data import Dataset
import os
import pickle
import gzip


# from common.utils.io_utils import load_pickle
# from pvn3d.utils.pc_utils import pad_or_clip_v2
# from pvn3d.utils.img_utils import normalize_image
# from pvn3d.utils.o3d_utils import cal_normals


def load_pickle(path):
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.pgz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise RuntimeError('Unsupported extension {}.'.format(os.path.splitext(path)[-1]))


def dump_pickle(obj, path):
    if path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    elif path.endswith('.pgz'):
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        raise RuntimeError('Unsupported extension {}.'.format(os.path.splitext(path)[-1]))


def print_dict(d: dict):
    """Print the given dictionary for debugging."""
    for k, v in d.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, v.shape)
        else:
            print(k, v)


def normalize_image(img):
    return (img.astype(np.float) - 127.5) / 127.5


def pad_or_clip_v2(array: np.array, n: int):
    if array.shape[0] >= n:
        return array[:n]
    else:
        pad = np.repeat(array[0:1], n - array.shape[0], axis=0)
        return np.concatenate([array, pad], axis=0)


def cal_normals(points, camera_location=np.array([0., 0., 0.])):
    cloud = o3d.geometry.PointCloud()
    cld = points.astype(np.float32)
    cloud.points = o3d.utility.Vector3dVector(cld)
    cloud.estimate_normals()
    cloud.orient_normals_towards_camera_location(camera_location)
    n = np.array(cloud.normals).copy()
    return n


def project_points(points, cam_intrinsic):
    p2d = np.dot(points, cam_intrinsic.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d


def draw_points(img, points, r=1, color=(255, 0, 0)):
    h, w = img.shape[0], img.shape[1]
    for pt_2d in points:
        pt_2d[0] = np.clip(pt_2d[0], 0, w)
        pt_2d[1] = np.clip(pt_2d[1], 0, h)
        img = cv2.circle(
            img, (pt_2d[0], pt_2d[1]), r, color, -1
        )
    return img


class TOCDataset(Dataset):
    _fp32_fields = ('color_image', 'points', 'feature', 'normals', 'RT_arr', 'center_arr', 'keypoints_arr',
                    'center_targ_offset', 'keypoints_targ_offset', 'cam_intrinsic', 'cam_extrinsic')
    _int64_fields = ('labels', 'fg_labels', 'choose_pixel')
    _bool_fields = ('points_mask',)

    def __init__(self, dataset_name,
                 root_dir,
                 voxel_size=None,
                 shuffle_points=False,
                 num_points=8192,
                 num_keypoints=8,
                 num_objects=7,
                 to_tensor=True,
                 remove_table=False):
        # TODO: add data augmentation
        self.dataset_name = dataset_name
        self.root_dir = Path(root_dir)
        self.point_cloud_dir = self.root_dir / 'object_point_clouds'
        if not self.root_dir.exists():
            logger.error(f'Not exists root dir: {self.root_dir}')
        if not self.point_cloud_dir.exists():
            logger.error(f'Not exists point cloud dir: {self.point_cloud_dir}')

        self.voxel_size = voxel_size
        self.shuffle_points = shuffle_points
        self.num_points = num_points
        self.num_keypoints = num_keypoints
        self.num_objects = num_objects
        self.to_tensor = to_tensor
        self.remove_table = remove_table

        self.list_file = self.root_dir / f'{dataset_name}_list.txt'
        if not self.list_file.exists():
            logger.error(f'Not exists list file: {self.list_file}')
        assert self.list_file.exists(), f'{self.list_file} not exists.'
        self.data_list = self._gen_data_list()
        logger.info(f'TOC Dataset: name: {dataset_name}, '
                    f'length: {len(self.data_list)}, '
                    f'remove_table: {self.remove_table}')

    def _gen_data_list(self):
        data_list = []
        with open(self.list_file, 'r') as f:
            for l in f.readlines():
                l = l.strip()
                path_list = {
                    'scene_id': l,
                    'color': self.root_dir / f'{l}_color_kinect.png',
                    'depth': self.root_dir / f'{l}_depth_kinect.png',
                    'seg_label': self.root_dir / f'{l}_label_kinect.png',
                    'meta': self.root_dir / f'{l}_meta.pkl',
                }
                all_exists = True
                for v in path_list.values():
                    if isinstance(v, str):
                        continue
                    if not v.exists():
                        all_exists = False
                        break
                if all_exists:
                    data_list.append(path_list)
        return data_list

    def __getitem__(self, index):
        out_dict = {}
        path_list = self.data_list[index]

        # Load images
        color_image = cv2.imread(str(path_list['color']), cv2.IMREAD_COLOR)
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # color_image = normalize_image(color_image)
        depth_image = cv2.imread(str(path_list['depth']), cv2.IMREAD_UNCHANGED)
        label_image = cv2.imread(str(path_list['seg_label']), cv2.IMREAD_UNCHANGED)

        # Load meta info
        meta_info = load_pickle(str(path_list['meta']))
        id2name = meta_info['id2name']
        valid_seg_ids = np.array(list(id2name.keys()))
        cam_intrinsic = meta_info['intrinsic']
        cam_extrinsic = meta_info['extrinsic']

        # unproject depth to 3d points
        v, u = np.indices(depth_image.shape)  # [H, W], [H, W]
        z = depth_image / 1000.0  # [H, W]
        uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
        points_viewer = uv1 @ np.linalg.inv(cam_intrinsic).T * z[..., None]  # [H, W, 3]

        # Mask out invalid points
        # valid_seg_mask = np.isin(label_image, valid_seg_ids)
        mask = np.logical_and(depth_image > 0, label_image != 1)  # ground label=1, table label=47
        if self.remove_table:
            table_id = 13
            mask = np.logical_and(mask, label_image != table_id)
        choose_pixel = mask.flatten().nonzero()[0].astype(np.int64)
        # mask = np.logical_and(depth_image > 0, valid_seg_mask)

        points = points_viewer[mask]  # [N, 3]
        colors = color_image[mask]  # [N, 3]
        labels = label_image[mask]  # [N]

        if self.voxel_size:
            sample_indices = voxel_down_sample(points, self.voxel_size)
            points = points[sample_indices]
            colors = colors[sample_indices]
            labels = labels[sample_indices]
            choose_pixel = choose_pixel[sample_indices]

        if self.shuffle_points:
            shuffle_indices = np.random.permutation(points.shape[0])
            points = points[shuffle_indices]
            colors = colors[shuffle_indices]
            labels = labels[shuffle_indices]
            choose_pixel = choose_pixel[shuffle_indices]

        if self.num_points:
            points = pad_or_clip_v2(points, self.num_points)
            colors = pad_or_clip_v2(colors, self.num_points)
            labels = pad_or_clip_v2(labels, self.num_points)
            choose_pixel = pad_or_clip_v2(choose_pixel, self.num_points)

        fg_labels = np.isin(labels, valid_seg_ids)
        labels[np.logical_not(np.isin(labels, valid_seg_ids))] = 1  # # set all unseen objects to ground class
        labels[labels > self.num_objects] = 1  # original label starts from 1 and 1 is ground class

        RT_arr = np.zeros((self.num_objects, 3, 4))
        center_arr = np.zeros((self.num_objects, 3))
        keypoints_arr = np.zeros((self.num_objects, self.num_keypoints, 3))
        center_targ_offset = np.zeros((self.num_points, 3))
        keypoints_targ_offset = np.zeros((self.num_points, self.num_keypoints, 3))
        for object_id in range(1, self.num_objects + 1):
            if object_id not in meta_info['id2name'].keys():
                continue
            object_name = meta_info['id2name'][object_id]
            scale = meta_info['scales'][object_id]
            RT = meta_info['poses'][object_id]  # pose in world frame
            RT = np.matmul(cam_extrinsic, RT)  # pose in camera frame

            RT_arr[object_id - 1] = RT[:3, :]
            r = RT[:3, :3]
            t = RT[:3, 3]

            center = np.loadtxt(self.point_cloud_dir / f'{object_name}_center.txt')
            center = np.dot(center, r.T) + t
            center_arr[object_id - 1] = center

            object_mask = (labels == object_id)
            target_offset = points - center
            center_targ_offset[object_mask] = target_offset[object_mask]

            keypoints = np.loadtxt(self.point_cloud_dir / f'{object_name}_key_points.txt')
            keypoints *= scale[np.newaxis, :]
            keypoints = np.dot(keypoints, r.T) + t
            keypoints = keypoints[:self.num_keypoints]
            keypoints_arr[object_id - 1] = keypoints
            target_offset = points[:, np.newaxis, :] - keypoints[np.newaxis, ...]
            keypoints_targ_offset[object_mask] = target_offset[object_mask]

        normals = cal_normals(points)
        # set background label to 0
        labels -= 1  # set labels starts from zero

        out_dict['color_image'] = color_image.transpose(2, 0, 1)
        out_dict['points'] = points
        out_dict['choose_pixel'] = choose_pixel
        out_dict['feature'] = colors
        out_dict['fg_labels'] = fg_labels
        out_dict['labels'] = labels
        out_dict['normals'] = normals
        out_dict['RT_arr'] = RT_arr
        out_dict['center_arr'] = center_arr
        out_dict['keypoints_arr'] = keypoints_arr
        out_dict['center_targ_offset'] = center_targ_offset
        out_dict['keypoints_targ_offset'] = keypoints_targ_offset
        out_dict['cam_intrinsic'] = cam_intrinsic
        out_dict['cam_extrinsic'] = cam_extrinsic
        out_dict['scene_id'] = path_list['scene_id']

        if self.to_tensor:
            out_dict = self.convert_to_tensor(out_dict)
        cld_rgb_nrm = torch.cat([out_dict['points'],
                                 out_dict['feature'],
                                 out_dict['normals']], dim=1)
        choose_pixel = out_dict['choose_pixel'].unsqueeze(0)
        class_ids = torch.tensor(np.arange(self.num_objects)).view(self.num_objects, 1).contiguous()
        out_dict['cld_rgb_nrm'] = cld_rgb_nrm
        out_dict['class_ids'] = class_ids
        # return out_dict
        if self.dataset_name == 'test':
            return out_dict['color_image'], \
                   out_dict['points'], \
                   out_dict['cld_rgb_nrm'], \
                   out_dict['choose_pixel'].unsqueeze(0), \
                   out_dict['keypoints_targ_offset'], \
                   out_dict['center_targ_offset'], \
                   out_dict['class_ids'], \
                   out_dict['RT_arr'], \
                   out_dict['labels'], \
                   out_dict['keypoints_arr'], \
                   out_dict['center_arr'], \
                   out_dict['scene_id']
        else:
            return out_dict['color_image'], \
                   out_dict['points'], \
                   out_dict['cld_rgb_nrm'], \
                   out_dict['choose_pixel'].unsqueeze(0), \
                   out_dict['keypoints_targ_offset'], \
                   out_dict['center_targ_offset'], \
                   out_dict['class_ids'], \
                   out_dict['RT_arr'], \
                   out_dict['labels'], \
                   out_dict['keypoints_arr'], \
                   out_dict['center_arr'], \
                   out_dict['scene_id']

    def __len__(self):
        return len(self.data_list)

    def convert_to_tensor(self, input_dict):
        output_dict = dict()
        for key, value in input_dict.items():
            if key in self._fp32_fields:
                output_dict[key] = torch.tensor(value, dtype=torch.float32)
            elif key in self._int64_fields:
                output_dict[key] = torch.tensor(value, dtype=torch.int64)
            elif key in self._bool_fields:
                output_dict[key] = torch.tensor(value, dtype=torch.bool)
            else:
                # logger.warning('Field({}) is not converted to tensor.'.format(key))
                output_dict[key] = value
        return output_dict


def voxel_down_sample(points, voxel_size, min_bound=(-5.0, -5.0, -5.0), max_bound=(5.0, 5.0, 5.0)):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsample_pcd, mapping, index_buckets = pcd.voxel_down_sample_and_trace(
        voxel_size, np.array(min_bound)[:, None], np.array(max_bound)[:, None])
    sample_indices = [int(x[0]) for x in index_buckets]
    return sample_indices


def main():
    import open3d
    import matplotlib.pyplot as plt
    cm = plt.get_cmap('jet')

    dataset = TOCDataset(dataset_name='test',
                         # root_dir='/home/xulab/Downloads/evaluation_data',
                         root_dir='/media/shanaf/HDD2/Songfang/DATA/PVN3D/v2_scene1',
                         voxel_size=0.001,
                         shuffle_points=True,
                         num_objects=16,
                         remove_table=True)
    data_idx = 0
    data = dataset[data_idx]
    # visualize_point_cloud(data['points'], colors=data['colors'], show_frame=True)
    # visualize_point_cloud(data['points'], colors=data['labels'][:, None] * (1.0, 0.0, 0.0) + (0.0, 0.0, 1.0),
    #                       show_frame=True)
    print_dict(data)
    color_image = data['color_image'].numpy()
    points = data['points'].numpy()
    feature = data['feature'].numpy()
    labels = data['labels'].numpy()
    normals = data['normals'].numpy()
    center_arr = data['center_arr'].numpy()
    keypoints_arr = data['keypoints_arr'].numpy()
    cam_intrinsic = data['cam_intrinsic'].numpy()
    cam_extrinsic = data['cam_extrinsic'].numpy()
    center_targ_offset = data['center_targ_offset'].numpy()
    keypoints_targ_offset = data['keypoints_targ_offset'].numpy()

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(feature / 2 + 0.5)
    pcd.normals = open3d.utility.Vector3dVector(normals)
    open3d.visualization.draw_geometries([pcd], window_name=f'{data_idx}-pointcloud')

    color = cm(labels / dataset.num_objects)[:, :3]
    pcd.colors = open3d.utility.Vector3dVector(color)
    open3d.visualization.draw_geometries([pcd], window_name=f'{data_idx}-segment')
    pcd.colors = open3d.utility.Vector3dVector(feature / 2 + 0.5)

    color_image = color_image.transpose(1, 2, 0)  # [...,::-1].copy()
    color_image = (color_image * 127.5 + 127.5).astype(np.uint8)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(f'{data_idx}-RGB', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for obj_idx in range(dataset.num_objects):
        kp3d = keypoints_arr[obj_idx]
        if kp3d.sum() < 1e-6:
            continue
        kp_2ds = project_points(kp3d, cam_intrinsic)
        color_image = draw_points(color_image, kp_2ds, 3, (255, 0, 0))
        ctr3d = center_arr[obj_idx]
        ctr_2ds = project_points(ctr3d[None, :], cam_intrinsic)
        color_image = draw_points(color_image, ctr_2ds, 4, (0, 0, 255))
        kpcd = open3d.geometry.LineSet()
        kpcd.points = open3d.utility.Vector3dVector(keypoints_arr[obj_idx])
        line_idx = []
        for i in range(keypoints_arr.shape[1] - 1):
            line_idx.append([i, i + 1])
        line_idx = np.stack(line_idx).astype(np.int)
        kpcd.lines = open3d.utility.Vector2iVector(line_idx)
        kpcd_colors = np.zeros((keypoints_arr.shape[1], 3))
        kpcd_colors[:, 0] = 1.0
        kpcd.colors = open3d.utility.Vector3dVector(kpcd_colors)

        center = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        center.paint_uniform_color(np.array([[0], [0], [1]]))
        center.translate(center_arr[obj_idx][..., np.newaxis])

        point_mask = labels == obj_idx
        if point_mask.sum() > 10:
            point_mask = np.random.choice(point_mask.nonzero()[0], 10, replace=False)
        choose_points = points[point_mask]
        choose_center_offset = center_targ_offset[point_mask]
        choose_keypoints_offset = keypoints_targ_offset[point_mask]
        num_points = choose_points.shape[0]
        center_offset = choose_points - choose_center_offset
        center_offset_linset = open3d.geometry.LineSet()
        center_offset_linset.points = open3d.utility.Vector3dVector(
            np.concatenate([choose_points, center_offset], axis=0))
        line_idx = []
        for i in range(num_points):
            line_idx.append([i, i + num_points])
        line_idx = np.stack(line_idx).astype(np.int)
        center_offset_linset.lines = open3d.utility.Vector2iVector(line_idx)
        colors = np.zeros((num_points, 3))
        colors[:, 2] = 1.0
        center_offset_linset.colors = open3d.utility.Vector3dVector(colors)

        open3d.visualization.draw_geometries([pcd, kpcd, center, center_offset_linset],
                                             window_name=f'{data_idx}-{obj_idx}')

    cv2.imshow(f'{data_idx}-pts', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
