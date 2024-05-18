from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from .dafdetr import DAFDeTr


@DETECTORS.register_module()
class DAFDeTrWaymo(DAFDeTr):
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
        super(DAFDeTrWaymo, self).__init__(
            use_img=use_img,
            freeze_img=freeze_img,
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_voxel_layer=pts_voxel_layer,
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

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
        img_feats, point_feats = self.extract_feat(img, points, img_metas)
        bbox_list = self.bbox_head.simple_test_bboxes(img_feats,
                                                      point_feats,
                                                      img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
