from loguru import logger
import numpy as np

import torch

from lib.utils.meanshift_pytorch import MeanShiftTorch


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions

    Args:
        A: np.array, (npts, 3), usually points in model space
        B: np.array, (npts, 3), usually points in camera space

    Returns:
        T: np.array, (4, 4), homogeneous transformation matrix that maps A on to B
        R: np.array, (3, 3), rotation matrix
        t: np.array, (3, 1), translation vector

    """
    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T


def get_pose_from_voting(voted_key_pts,
                         voted_center,
                         gt_key_pts,
                         gt_center,
                         use_center=False,
                         use_center_cluster_filter=False,
                         radius=0.08):
    """

    Args:
        voted_key_pts: torch.tensor, (num_kpts, num_pts, 3)
        voted_center: torch.tensor, (num_pts, 3)
        gt_key_pts: torch.tensor, (num_kpts, 3)
        gt_center: torch.tensor, (1, 3)
        use_center: bool, take center into PnP solver
        use_center_cluster_filter: bool, filter key point with center clustering
        radius: float

    Returns:
        pred_pose: np.array, (3, 4)
    """
    num_key_pts = voted_key_pts.shape[0]
    if use_center:
        cluster_kps = torch.zeros(num_key_pts + 1, 3, device=voted_key_pts.device)
    else:
        cluster_kps = torch.zeros(num_key_pts, 3, device=voted_key_pts.device)

    # Meanshift clustering to calculate key points and center from voting
    ms = MeanShiftTorch(bandwidth=radius)
    ctr, ctr_labels = ms.fit(voted_center)
    if ctr_labels.sum() < 1:
        ctr_labels[0] = 1
    if use_center:
        cluster_kps[num_key_pts, :] = ctr

    if use_center_cluster_filter:
        in_pred_key_pts = voted_key_pts[:, ctr_labels, :]
    else:
        in_pred_key_pts = voted_key_pts

    # TODO: batch operation?
    for ikp, kps3d in enumerate(in_pred_key_pts):
        cluster_kps[ikp, :], _ = ms.fit(kps3d)

    # mesh_kps = bs_utils_lm.get_kps(obj_id, ds_type="linemod")
    if use_center:
        mesh_kps = np.concatenate([gt_key_pts, gt_center.reshape(1, 3)], axis=0)
    else:
        mesh_kps = gt_key_pts

    mesh_kps = torch.tensor(mesh_kps).float().cuda()
    pred_pose = best_fit_transform(mesh_kps.contiguous().detach().cpu().numpy(),
                                   cluster_kps.contiguous().detach().cpu().numpy())

    return pred_pose, cluster_kps


def get_pred_poses(data_batch,
                   preds,
                   use_center=True,
                   use_center_cluster_filter=False,
                   save_cluster_keypoints=True):
    """
    Compute object pose from prediction.
    1. Meanshift clustering algorithm to find keypoint and center point
    2. PnP solver to find the 6DPose from predicted keypoint in camera space and keypoint in object space.

    Args:
        data_batch: dict of data
        preds: dict of prediction from module
        use_center: bool, taking center point into PnP solver or not
        use_center_cluster_filter: bool, filter key points?

    Returns:
        batch_pred_pose_list: batch list of pred_pose,
             - pred_pose: np.array, (4, 4)
    """
    batch_size = data_batch['points'].shape[0]
    num_objects, num_keypoints = data_batch['keypoints_arr'].shape[1], data_batch['keypoints_arr'].shape[2]
    batch_pred_pose_list = []
    if save_cluster_keypoints:
        preds['keypoints'] = torch.zeros(batch_size, num_objects, num_keypoints, 3, device=data_batch['points'].device)
        if use_center:
            preds['centers'] = torch.zeros(batch_size, num_objects, 3, device=data_batch['points'].device)
    for batch_idx in range(batch_size):
        points = data_batch['points'][batch_idx].detach().clone()  # npts, 3
        gt_key_pts_viewer = data_batch['keypoints_arr'][batch_idx].detach().clone()  # num_object, num_key_pts, 3
        gt_center_viewer = data_batch['center_arr'][batch_idx].detach().clone()  # num_object , 3
        RT_arr = data_batch['RT_arr'][batch_idx].detach().clone()  # num_object, 3, 4

        pred_kpts_offset = preds['keypoints_pred_offset'][batch_idx].detach().clone()  # npts, num_key_pts, 3
        # pred_kpts_offset = pred_kpts_offset.permute(1, 0, 2)  # num_key_pts, npts, 3
        pred_center_offset = preds['center_pred_offset'][batch_idx].detach().clone()  # npts, 1, 3
        # logger.info(f'pred_center_offset: {pred_center_offset.shape}')
        # pred_center_offset = pred_center_offset.permute(1, 0, 2)  # 1, npts, 3
        pred_seg_logits = preds['seg_logits'][batch_idx].detach().clone()  # num_object, npts
        _, pred_label = torch.max(pred_seg_logits, dim=1)  # npts
        pred_label = pred_label.detach().clone()

        num_key_pts, n_pts, _ = pred_kpts_offset.size()
        pred_center = points - pred_center_offset[0]
        # logger.info(f'{points.shape}, {num_key_pts}, {n_pts}')
        # logger.info(f'pred_kpts:offset: {pred_kpts_offset.shape}')
        # exit()
        pred_key_pts = points.view(1, n_pts, 3).repeat(num_key_pts, 1, 1) - pred_kpts_offset

        pred_pose_list = {}
        for class_id in range(num_objects):
            if class_id == 0:
                logger.info(f'Ignore pose for class_id {class_id}')
                continue
            class_mask = pred_label == class_id
            if class_mask.sum() < 1:
                logger.info(f'Detect no label for class_id {class_id}, return identity pose')
                # to be consistent with annotation
                pred_pose_list[class_id + 1] = np.identity(4)
            else:
                gt_key_pts_one_object = gt_key_pts_viewer[class_id].detach().cpu().numpy()
                gt_center_one_object = gt_center_viewer[class_id].detach().cpu().numpy()
                rt_one_object = RT_arr[class_id].detach().cpu().numpy()

                gt_key_pts_obj_space = np.matmul(gt_key_pts_one_object - rt_one_object[:3, 3],
                                                 rt_one_object[:3, :3])
                gt_center_obj_space = np.matmul(gt_center_one_object - rt_one_object[:3, 3],
                                                rt_one_object[:3, :3]).reshape(1, 3)

                class_voted_kps = pred_key_pts[:, class_mask, :]
                class_voted_center = pred_center[class_mask, :]
                pred_pose, cluster_keypoints = get_pose_from_voting(voted_key_pts=class_voted_kps,
                                                                    voted_center=class_voted_center,
                                                                    gt_key_pts=gt_key_pts_obj_space,
                                                                    gt_center=gt_center_obj_space,
                                                                    use_center=use_center,
                                                                    use_center_cluster_filter=use_center_cluster_filter)
                if save_cluster_keypoints:
                    preds['keypoints'][batch_idx, class_id] = cluster_keypoints[:num_keypoints, :]
                    if use_center:
                        preds['centers'][batch_idx, class_id] = cluster_keypoints[num_keypoints, :]
                pred_pose_4 = np.eye(4)
                pred_pose_4[:3, :] = pred_pose
                logger.info(f'Detect prediction pose for class {class_id}')
                # to be consistent with annotation
                pred_pose_list[class_id + 1] = pred_pose_4

        batch_pred_pose_list.append({'poses': pred_pose_list})

    return batch_pred_pose_list