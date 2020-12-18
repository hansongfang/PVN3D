from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
from pathlib import Path
from loguru import logger
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
# from common import Config
from lib import PVN3D
from datasets.toc.toc_dataset import TOCDataset
# from datasets.linemod.linemod_dataset import LM_Dataset
from lib.utils.sync_batchnorm import convert_model
# from lib.utils.pvn3d_eval_utils import cal_frame_poses, cal_frame_poses_lm
# from lib.utils.basic_utils import Basic_Utils
from cv2 import imshow, waitKey
from post_process import get_pred_poses


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
args = parser.parse_args()


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        try:
            checkpoint = torch.load(filename)
        except:
            checkpoint = pkl.load(open(filename, "rb"))
        epoch = checkpoint["epoch"]
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def print_dict(d: dict):
    """Print the given dictionary for debugging."""
    for k, v in d.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, v.shape)
        else:
            print(k, v)


def cal_view_pred_pose(model, data):
    """calculate the prediction pose """
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = [item.to("cuda", non_blocking=True) for item in data]
        rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, ctr_targ_ofst, \
            cls_ids, rts, labels, kp_3ds, ctr_3ds = cu_dt

    pred_kp_of, pred_rgbd_seg, pred_ctr_of = model(
        cld_rgb_nrm, rgb, choose
    )
    _, classes_rgbd = torch.max(pred_rgbd_seg, -1)

    preds = {}
    preds['keypoints_pred_offset'] = pred_kp_of
    preds['seg_logits'] = pred_rgbd_seg
    preds['center_pred_offset'] = pred_ctr_of
    print_dict(preds)

    data_batch = {}
    data_batch['points'] = pcld
    data_batch['RT_arr'] = rts
    data_batch['keypoints_arr'] = kp_3ds
    data_batch['center_arr'] = ctr_3ds
    print_dict(data_batch)

    batch_pred_pose_list = get_pred_poses(data_batch,
                                          preds)
    return batch_pred_pose_list


def main():
    batch_size = 1
    num_objects = num_class = 12
    n_sample_points = 8192
    root_dir = '/media/shanaf/HDD2/Songfang/DATA/PVN3D/v2_scene1'
    model_weight_file = './train_log/toc/checkpoints/level1_pvn3d_best.pth.tar'
    voxel_size = None
    shuffle_points = True,
    remove_table = True
    test_ds = TOCDataset('test',
                          root_dir=root_dir,
                          voxel_size=None,
                          shuffle_points=shuffle_points,
                          num_objects=num_objects,
                          remove_table=remove_table)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    model = PVN3D(
        num_classes=num_class,
        pcld_input_channels=6,
        pcld_use_xyz=True,
        num_points=n_sample_points
    ).cuda()
    model = convert_model(model)
    model.cuda()

    # load status from checkpoint
    # if args.checkpoint is not None:
    if os.path.exists(model_weight_file):
        checkpoint_status = load_checkpoint(
            model, None, filename=model_weight_file[:-8]
        )
    model = nn.DataParallel(model)

    for i, data in tqdm.tqdm(
        enumerate(test_loader), leave=False, desc="val"
    ):
        scene_id_list = data[-1]
        data = data[:-1]

        batch_pred_pose_list = cal_view_pred_pose(model, data)

        import pickle as pkl
        pose_output_dir = Path(model_weight_file).parent / "pose_pred_results"
        pose_output_dir.mkdir(exist_ok=True, parents=True)
        for idx, scene_id in enumerate(scene_id_list):
            curr_path = pose_output_dir / f"{scene_id}_pred.pkl"
            with open(curr_path, "wb") as handle:
                pkl.dump(batch_pred_pose_list[idx], handle, protocol=pkl.HIGHEST_PROTOCOL)
            logger.info(f"{scene_id} finish. Save to {curr_path}")


if __name__ == "__main__":
    main()