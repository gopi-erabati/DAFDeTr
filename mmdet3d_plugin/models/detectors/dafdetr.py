import time
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.core import bbox3d2result
from mmdet3d.models import build_backbone, build_head, build_neck, DETECTORS
# from mmdet.models import DETECTORS
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.models.builder import (build_voxel_encoder,
                                    build_middle_encoder)
from mmdet3d.ops import Voxelization


@DETECTORS.register_module()
class DAFDeTr(Base3DDetector):
    """
    DAFDetr: Deformable Attention Fusion Based 3D Detection Transformer.
    Soft association and sequential fusion of LiDAR and Camera features
    leveraging deformable cross-attention
    """

    def __init__(self,
                 use_img=False,
                 freeze_img=True,
                 img_backbone=None,
                 img_neck=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DAFDeTr, self).__init__(init_cfg)
        self.use_img = use_img
        self.freeze_img = freeze_img

        self.img_backbone = None
        self.img_neck = None
        if self.use_img:
            # IMAGE FEATURES : BACKBONE + NECK
            # build backbones (img)
            if img_backbone is not None:
                self.img_backbone = build_backbone(img_backbone)

            # build neck (img)
            if img_neck is not None:
                self.img_neck = build_neck(img_neck)

        # POINTS FEATURES : Points Voxel Layer, Points Voxel Encoder,
        # Points Voxel Scatter, Pts backbone (SECOND), Pts neck (FPN)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)
        else:
            self.pts_neck = None

        # build head
        bbox_head.update(train_cfg=train_cfg.pts if train_cfg is not None
        else None)
        bbox_head.update(test_cfg=test_cfg.pts)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.time_list = []

    def init_weights(self):

        super(DAFDeTr, self).init_weights()

        if self.freeze_img:
            if self.img_backbone is not None:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.img_neck is not None:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, points, img=None, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(img, points, **kwargs)
        else:
            return self.forward_test(img, points, **kwargs)

    def forward_train(self,
                      img,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas,
                      gt_bboxes_3d_ignore=None):
        """
        Args:
            img (Tesnor): Input RGB image of shape (N, C, H, W)
            proj_img (Tensor): Projected LiDAR point cloud of shape (N, C,
                H, W)
            proj_points (Tensor): Projected LiDAR 3D Points (B, N, 3)
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other detailssee
                :class `mmdet3d.datasets.pipelines.Collect`.
            proj_idxs (List[Tensor]): A list of tensors containing the
                indexes of projected point cloud onto 'spherical' or 'BEV'
                images corresponding to their 2D indexes on RGB image.
                Shape (size(points3d), 2)
            img_idxs (List[Tensor]): A list of tensors containing indexes of
                projected 3D points onto 2D RGB image.
                Shape (size(points3d), 2)
            gt_bboxes_3d (List[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes of shape (num_box, 7)
            gt_labels_3d (List[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes labels
                of shape (num_box, )
            gt_bboxes_3d_ignore (None | list[Tensor]): Specify which
                bounding boxes can be ignored when computing the loss.

        Returns:
            dict [str, Tensor]: A dictionary of loss components
        """
        img_feats, point_feats = self.extract_feat(img, points, img_metas)
        # list[(B * N, 128, H, W)], # [(B, 128, H, W), ...]
        # LiDAR: H, W is 8, 16, 32, 64 stride of 1472
        # Img of strides 4, 8, 16, 32

        losses = self.bbox_head.forward_train(img_feats, point_feats,
                                              gt_bboxes_3d, gt_labels_3d,
                                              gt_bboxes_3d_ignore, img_metas)
        return losses

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, points, img_metas=None):
        """Extract Image and Point Features"""

        # Image Features
        if self.use_img:
            img_feats = self.extract_img_feat(img, img_metas)  # list[Tensor]
        else:
            img_feats = None
        # list[(B * N, C, H, W)...]

        # Point Features
        point_feats = self.extract_point_features(points)  # list[Tensor]
        # [(B, 128, H, W), ...] H, W is 8, 16, 32, 64 stride of 1472

        return img_feats, point_feats
        # list[(B, N, C, H, W)], # [(B, 128, H, W), ...]
        # LiDAR: H, W is 8, 16, 32, 64 stride of 1472
        # Img of strides 4, 8, 16, 32

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.img_neck is not None:
            img_feats = self.img_neck(img_feats)
        return list(img_feats)  # list[(B*N, C, H, W)...]

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_point_features(self, points):
        """ Extract features of Points using encoder, middle encoder,
        backbone and neck.
        Here points is list[Tensor] of batch """

        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.pts_neck is not None:
            x = self.pts_neck(x)
        return x
        # [(B, 128, H, W), ...] H, W is 8, 16, 32, 64 stride of 1472

    def forward_test(self,
                     img,
                     points,
                     img_metas,
                     **kwargs):
        """
        Args:
            img (list[torch.Tensor]): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch.
            proj_img (list[torch.Tensor]): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all lidar projected images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            proj_idxs (list[list[torch.Tensor]]): the outer list indicates
                test time augmentation and inner list includes all the
                projected lidar points indexes on image in a batch. Inner list
                length of batch size with each tensor shape (num_pts, 2)
            img_idxs (list[list[torch.Tensor]]): the outer list indicates
                test time augmentation and inner list includes all the
                image indexes for each projected cloud in a batch. Inner list
                length of batch size with each tensor shape (num_pts, 2)
        """
        # for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
        #     if not isinstance(var, list):
        #         raise TypeError('{} must be a list, but got {}'.format(
        #             name, type(var)))
        #
        # num_augs = len(img)
        # if num_augs != len(img_metas):
        #     raise ValueError(
        #         'num of augmentations ({}) != num of image meta ({})'.format(
        #             len(img), len(img_metas)))
        #
        # return self.simple_test(img[0], points[0], img_metas[0], **kwargs)

        if img is not None:
            return self.simple_test(img[0], points[0], img_metas[0], **kwargs)
        else:
            return self.simple_test(img, points[0], img_metas[0], **kwargs)

    def simple_test(self, img, points, img_metas, rescale=False):
        """ Test function without test-time augmentation.

        Args:
            img (Tesnor): Input RGB image of shape (N, C, H, W)
            proj_img (Tensor): Projected LiDAR point cloud of shape (N, C,
                H, W)
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other detailssee
                :class `mmdet3d.datasets.pipelines.Collect`.
            proj_idxs (Tensor): A tesnor containing the indexes of projected
                point cloud onto 'spherical' or 'BEV' images correspodning to
                their 2D indexes on RGB image.
                Shape (N, size(points3d), size(points3d))
            img_idxs (Tesnor): A tesnor containing indexes of projected 3D
                points onto 2D RGB image.
                Shape (N, size(points3d), size(points3d))

        Returns:
            list[dict]: Predicted 3d boxes. Each list consists of a dict
            with keys: boxes_3d, scores_3d, labels_3d.
        """
        # st = time.time()
        img_feats, point_feats = self.extract_feat(img, points, img_metas)


        # # code to drop some image features
        # num_drop_imgs = 4
        # import random
        # drop_img_idxs = random.sample([i for i in range(6)], num_drop_imgs)
        # # list[idx, ]
        # for feat_idx, img_feat in enumerate(img_feats):
        #     img_feats[feat_idx][drop_img_idxs] = 0.0

        bbox_list = self.bbox_head.simple_test_bboxes(img_feats,
                                                      point_feats,
                                                      img_metas)
        # et = time.time()
        # self.time_list.append(et-st)
        # if len(self.time_list) > 1100:
        #     print(f"avg time: {np.mean(self.time_list[100:1099])}")
            # np.save("/home/gopi/PhD/workdirs/RetFormer/waymo"
            #         "/retformer_waymo_D1_2x_3class/timearray.npy", np.array(
            #     self.time_list))

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test(self, img, proj_img, proj_idxs, img_idxs, img_metas,
                 rescale=False):
        pass
