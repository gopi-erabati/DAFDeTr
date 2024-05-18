import copy
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, normal_

from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        AssignResult)
from mmdet3d.models import HEADS, build_loss
# from mmdet.models import HEADS, build_loss

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.utils import build_transformer

from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core import (PseudoSampler, circle_nms, xywhr2xyxyr)
from mmcv.ops import nms_bev, MultiScaleDeformableAttention
from ...core.utils import (draw_heatmap_gaussian, gaussian_radius)
from ..utils.clip_sigmoid import clip_sigmoid


@HEADS.register_module()
class DAFDeTrHead(BaseDenseHead):
    """
    This is the head for DETR3D type detector with multi-sensor fusion (RGB +
    LiDAR)
    """

    def __init__(self,
                 with_img=False,
                 num_views=0,
                 in_channels_img=64,
                 lidar_feat_lvls=4,
                 img_feat_lvls=4,
                 num_proposals=128,
                 auxiliary=True,
                 in_channels_lidar=128 * 3,
                 hidden_channel=128,
                 num_classes=4,
                 with_encoder=False,
                 encoder_lidar=None,
                 # config for Transformer
                 transformer=None,
                 learnable_query_pos=False,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 num_heads=8,
                 ffn_channel=256,  # in TF
                 dropout=0.1,  # in TF
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_iou=dict(type='VarifocalLoss', use_sigmoid=True,
                               iou_weighted=True, reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 init_cfg=None,
                 pretrained=None,
                 ):
        super(DAFDeTrHead, self).__init__(init_cfg)

        self.initialize_by_heatmap = initialize_by_heatmap
        self.num_proposals = num_proposals
        self.with_img = with_img
        self.num_views = num_views
        self.nms_kernel_size = nms_kernel_size
        self.num_decoder_layers = transformer['decoder']['num_layers']
        self.bn_momentum = bn_momentum
        self.auxiliary = auxiliary
        self.num_classes = num_classes
        self.with_encoder = with_encoder
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        # Conv layer for LiDAR feats for neck as SECFPN
        # if not self.with_encoder:
        #     self.lidar_convs = nn.ModuleList()
        #     for _ in range(lidar_feat_lvls):
        #         self.lidar_convs.append(build_conv_layer(
        #             dict(type='Conv2d'),
        #             in_channels_lidar,  # 256 * 2
        #             hidden_channel,  # 128
        #             kernel_size=3,
        #             padding=1,
        #             bias=bias,
        #         ))

        if self.with_encoder:
            self.encoder_lidar = build_transformer_layer_sequence(
                encoder_lidar)

            # BEV position Embeddings
            self.bev_pos_encoder_mlvl_embed = nn.ModuleList()
            for _ in range(lidar_feat_lvls):
                self.bev_pos_encoder_mlvl_embed.append(
                    PositionEmbeddingLearned(
                        2, hidden_channel))

            # BEV Level Embeddings
            self.bev_level_embeds = nn.Parameter(torch.Tensor(
                lidar_feat_lvls, hidden_channel))

        # Query Initialization
        if self.initialize_by_heatmap:
            layers = []
            layers.append(ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
            layers.append(build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
            self.heatmap_head_lidar = nn.Sequential(*layers)
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.query_feat = nn.Parameter(
                torch.randn(1, hidden_channel, self.num_proposals))
            self.query_pos = nn.Parameter(
                torch.rand([1, self.num_proposals, 2]),
                requires_grad=learnable_query_pos)

        # With Image Features
        if self.with_img:
            # conv to reduce chanls of feat map of img
            self.img_convs = nn.ModuleList()
            for _ in range(img_feat_lvls):
                self.img_convs.append(build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels_img,  # channel of img feature map
                    hidden_channel,  # 128
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                ))
            # Img to BEV Transformation for query init
            # FC layer to transform of height collapsed img feat
            self.fc_imgcollaspsed = nn.Sequential(
                *[nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)])
            # Transformer Layer to get BEV image feats
            # cross-attention only layers for projecting img feature onto BEV
            self.img_to_bev_tf = nn.ModuleList()
            for i in range(num_views):
                self.img_to_bev_tf.append(
                    TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout,
                        activation,
                        self_posembed=PositionEmbeddingLearned(2,
                                                               hidden_channel),
                        cross_posembed=PositionEmbeddingLearned(2,
                                                                hidden_channel),
                        cross_only=True,
                    ))
            # conv layers to get heatmap of BEV image
            self.heatmap_head_img = copy.deepcopy(self.heatmap_head_lidar)

            # Prediction Head for LiDAR when using images
            self.prediction_heads_img = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                heads = copy.deepcopy(common_heads)
                heads.update(
                    dict(heatmap=(self.num_classes, num_heatmap_convs)))
                self.prediction_heads_img.append(
                    PredHeads(2 * hidden_channel, heads, conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg, bias=bias))

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        self.bev_pos_mlvl = []
        for lvl in range(lidar_feat_lvls):
            self.bev_pos_mlvl.append(
                self.create_2D_grid(int(x_size / (2 ** lvl)),
                                    int(y_size / (2 ** lvl))))
        # self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

        # TRANSFORMER
        self.transformer = build_transformer(transformer)

        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                PredHeads(hidden_channel, heads, conv_cfg=conv_cfg,
                          norm_cfg=norm_cfg, bias=bias))

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False
        self._init_assigner_sampler()

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base  # (1, x_size * y_size, 2)

    def init_weights(self):
        # initialize transformer
        self.transformer.init_weights()
        if self.with_img:
            for m in self.img_to_bev_tf.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()
        if self.with_encoder:
            normal_(self.bev_level_embeds)
            for m in self.modules():
                if isinstance(m, MultiScaleDeformableAttention):
                    m.init_weights()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_train(self,
                      img_feats, point_feats,
                      gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None,
                      img_metas=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            img_feats (list[Tensor] | None): Image feats list of stride 4, 8,
                16, 32 of shape (B * N, 128, H, W)
            point_feats (list[Tensor]): Point feat list [(B, 128, H, W)...]
                strides 8, 16, 32, 64 of 1472
            gt_bboxes (list[Tensor]): Ground truth bboxes ,
                shape (num_gts, 7).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 7).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(img_feats, point_feats, img_metas)  # list(dict)
        loss_inputs = [gt_bboxes, gt_labels, outs]
        losses = self.loss(*loss_inputs)
        return losses

    def forward(self, img_feats, lidar_feats, img_metas):
        """
        Forward Function
        Args:
            img_feats (list[(Tensor)]): Features of imgs from backbone
                (B*N, C, H, W), strides 4, 8, 16, 32
            lidar_feats (list[Tensor]): BEV LiDAR feats of shape
                [(B, 128, H, W)...] strides 8, 16, 32, 64 of 1472
            img_metas (list): list of img_meta for each image

        Returns:
            result_dict (list[dict]): A list of result_dict where dict has keys
                'center': (BS, 2, n_dec * n_prop)
                'height': (BS, 1, n_dec * n_prop)
                'dim': (BS, 3, n_dec * n_prop)
                'rot': (BS, 2, n_dec * n_prop)
                'vel': (BS, 2, n_dec * n_prop)
                'heatmap': (BS, n_cls, n_dec * n_prop)
                'dense_heatmpa': (BS, n_cls, H, W) - for layer0
                'query_heatmap_score': (BS, n_cls, n_prop) - for layer0, n_prop
                    from the H*W values which are top for each class and max
        """
        batch_size = lidar_feats[0].shape[0]

        # repeat the BEV positions for all batches
        bev_pos_mlvl_bs = []
        for bev_pos_lvl in self.bev_pos_mlvl:
            bev_pos_lvl = bev_pos_lvl.repeat(batch_size, 1, 1).to(
                lidar_feats[0].device)  # (BS, H*W, 2)
            bev_pos_mlvl_bs.append(bev_pos_lvl)

        # get the lidar feats from lidar_conv
        # if not self.with_encoder:
        #     for idx, lidar_conv in enumerate(self.lidar_convs):
        #         lidar_feats[idx] = lidar_conv(lidar_feats[idx])
        #         # (BS, h_dim, H, W) hidden_dim=128
        if self.with_encoder:
            # Encoder: MS Deformable Attention for LiDAR features
            # get the BEV positions and embeddings of all levels with level
            # embed and also feats of all levels flatten
            bev_pos_encoder_mlvl_norm = []
            bev_pos_encoder_mlvl_embed = []
            bev_spatial_shape_mlvl = []
            lidar_feat_mlvl = []
            for idx, (bev_pos_lvl, lidar_feat) in enumerate(
                    zip(bev_pos_mlvl_bs, lidar_feats)):
                bev_pos_encoder_lvl_embed = self.bev_pos_encoder_mlvl_embed[
                    idx](bev_pos_lvl)  # (BS, h_dim, H*W)
                bev_pos_encoder_lvl_embed = \
                    bev_pos_encoder_lvl_embed.permute(0, 2, 1)
                # (BS, H*W, h_dim)
                bev_pos_encoder_lvl_embed = bev_pos_encoder_lvl_embed + \
                                            self.bev_level_embeds[idx].view(
                                                1, 1, -1)  # (BS, H*W, h_dim)
                bev_pos_encoder_mlvl_embed.append(bev_pos_encoder_lvl_embed)

                # LiDAR feats
                lidar_feat_bs, lidar_feat_dim, lidar_feat_h, lidar_feat_w = \
                    lidar_feat.shape
                bev_spatial_shape = (lidar_feat_h, lidar_feat_w)
                bev_spatial_shape_mlvl.append(bev_spatial_shape)
                lidar_feat = lidar_feat.flatten(2).permute(0, 2, 1)
                # (BS, H*W, h_dim)
                lidar_feat_mlvl.append(lidar_feat)

                # normalize bev_pos_encoder_lvl with lidar_feat_h and
                # lidar_feat_w to make them lie in [0, 1] for reference points
                bev_pos_encoder_lvl_norm = bev_pos_lvl.float()
                bev_pos_encoder_lvl_norm[..., 0] /= lidar_feat_h
                bev_pos_encoder_lvl_norm[..., 1] /= lidar_feat_w
                bev_pos_encoder_mlvl_norm.append(bev_pos_encoder_lvl_norm)

            # concatenate all levels
            lidar_feat_mlvl = torch.cat(lidar_feat_mlvl, dim=1)
            # (BS, lvl*H*W, h_dim)
            bev_pos_encoder_mlvl_norm = torch.cat(bev_pos_encoder_mlvl_norm,
                                                  dim=1)
            # (BS, lvl*H*W, 2) normalized
            # repeat the bev_pos_encoder_mlvl (reference points) for all levels
            bev_pos_encoder_mlvl_norm = \
                bev_pos_encoder_mlvl_norm.unsqueeze(2).repeat(1, 1,
                                                              len(lidar_feats),
                                                              1)
            # (BS, lvl*H*W, lvls, 2)  normalized for reference points
            bev_pos_encoder_mlvl_embed = torch.cat(
                bev_pos_encoder_mlvl_embed, dim=1)  # (BS, lvl*H*W, h_dim)
            bev_spatial_shape_mlvl_tensor = torch.as_tensor(
                bev_spatial_shape_mlvl, dtype=torch.long,
                device=lidar_feat_mlvl.device)  # (lvl, 2)
            bev_level_start_index = torch.cat(
                (bev_spatial_shape_mlvl_tensor.new_zeros(
                    (1,)),
                 bev_spatial_shape_mlvl_tensor.prod(1).cumsum(0)[
                 :-1]))  # (lvl, )

            # reshape according to encoder expectation
            lidar_feat_mlvl = lidar_feat_mlvl.permute(1, 0, 2)
            # (lvl*H*W, BS, h_dim)
            bev_pos_encoder_mlvl_embed = bev_pos_encoder_mlvl_embed.permute(
                1, 0, 2)
            # (lvl*H*W, BS, h_dim)
            lidar_feat_mlvl_encoder = self.encoder_lidar(
                query=lidar_feat_mlvl,
                key=None,
                value=None,
                query_pos=bev_pos_encoder_mlvl_embed,
                spatial_shapes=bev_spatial_shape_mlvl_tensor,
                reference_points=bev_pos_encoder_mlvl_norm,
                level_start_index=bev_level_start_index
            )
            # (lvl*H*W, BS, h_dim)

            # get the first feat level to initialize query by heatmap
            # bring back the shape of feature maps
            lidar_feat_mlvl_encoder_list = lidar_feat_mlvl_encoder.split(
                [H_ * W_ for H_, W_ in bev_spatial_shape_mlvl],
                dim=0)
            # [(H*W, BS, h_dim), (H*W, BS, h_dim), ...]
            lidar_feats = []
            for level, (H_, W_) in enumerate(bev_spatial_shape_mlvl):
                memory_point_fmap = lidar_feat_mlvl_encoder_list[
                    level].permute(
                    1, 2, 0).reshape(lidar_feat_bs, lidar_feat_dim, H_, W_)
                lidar_feats.append(memory_point_fmap)
                # this contains list [(bs, c, h, w), ... for levels]
            lidar_feat = lidar_feats[0]
        else:
            lidar_feat = lidar_feats[0]

        # Query and Query Pos for Transformer Decoder
        if self.initialize_by_heatmap:
            dense_heatmap_lidar = self.heatmap_head_lidar(lidar_feat)
            # (BS, n_cls, H, W)

            lidar_feat_flatten = lidar_feat.view(batch_size,
                                                 lidar_feat.shape[1],
                                                 -1)  # [BS, h_dim, H*W]
            # bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(
            #     lidar_feat.device)  # (BS, H*W, 2)
            bev_pos = bev_pos_mlvl_bs[0]  # (BS, H*W, 2)

            # If Image feats are used
            if self.with_img:
                # GET THE BEV FEATURES USING Deformable Attention
                # Image to BEV
                # convert img_feats chnls to hidden_chnls
                for feat_idx, img_feat in enumerate(img_feats):
                    img_feats[feat_idx] = self.img_convs[feat_idx](img_feat)
                img_feat_s4 = img_feats[0]
                # img_feat = self.img_conv(img_feat_s4)
                # [BS * n_views, hidden_dim, H, W]

                img_h, img_w, num_channel = img_feat_s4.shape[-2], \
                    img_feat_s4.shape[-1], \
                    img_feat_s4.shape[1]
                raw_img_feat = img_feat_s4.view(batch_size, self.num_views,
                                                num_channel, img_h,
                                                img_w).permute(0, 2, 3, 1, 4)
                # (BS, h_dim, H, n_views, W)
                img_feat = raw_img_feat.reshape(batch_size, num_channel, img_h,
                                                img_w * self.num_views)
                # (BS, h_dim, H, n_views*W)
                img_feat_collapsed = img_feat.max(2).values
                # (BS, h_dim, 1, n_views*W)
                img_feat_collapsed = self.fc_imgcollaspsed(
                    img_feat_collapsed).view(
                    batch_size, num_channel, img_w * self.num_views)
                # (BS, h_dim, n_views*W)

                # positional encoding for image guided query initialization
                if self.img_feat_collapsed_pos is None:
                    img_feat_collapsed_pos = self.img_feat_collapsed_pos = self.create_2D_grid(
                        1, img_feat_collapsed.shape[-1]).to(img_feat.device)
                else:
                    img_feat_collapsed_pos = self.img_feat_collapsed_pos

                bev_feat = lidar_feat_flatten
                for idx_view in range(self.num_views):
                    bev_feat = self.img_to_bev_tf[idx_view](bev_feat,
                                                            img_feat_collapsed[
                                                            ...,
                                                            img_w * idx_view:img_w * (
                                                                    idx_view + 1)],
                                                            bev_pos,
                                                            img_feat_collapsed_pos[
                                                            :,
                                                            img_w * idx_view:img_w * (
                                                                    idx_view + 1)])
                    # (BS, C, H*W)

                # generate heatmap for query init
                dense_heatmap_img = self.heatmap_head_img(
                    bev_feat.view(lidar_feat.shape))  # (BS, num_classes, H, W)
                heatmap = (dense_heatmap_lidar.detach().sigmoid() +
                           dense_heatmap_img.detach().sigmoid()) / 2
                # (BS, num_classes, H, W)
            else:
                # if not use_img use only LiDAR heatmap
                heatmap = dense_heatmap_lidar.detach().sigmoid()
                # (BS, num_classes, H, W)

            padding = self.nms_kernel_size // 2  # 3 // 2 = 1
            local_max = torch.zeros_like(heatmap)  # (BS, num_classes, H, W)
            local_max_inner = F.max_pool2d(heatmap,
                                           kernel_size=self.nms_kernel_size,
                                           stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = \
                local_max_inner  # (BS, n_cls, H-1, W-1)
            ## for Pedestrian & Traffic_cone in nuScenes
            if self.test_cfg['dataset'] == 'nuScenes':
                local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1,
                                                 stride=1, padding=0)
                local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1,
                                                 stride=1, padding=0)
            elif self.test_cfg['dataset'] == 'Waymo':
                # for Pedestrian & Cyclist in Waymo
                local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1,
                                                 stride=1, padding=0)
                local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1,
                                                 stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max)  # locating max values
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
            # [BS, num_classes, H*W]

            # top #num_proposals among all classes
            top_proposals = heatmap.view(batch_size, -1).argsort(
                dim=-1, descending=True)[..., :self.num_proposals]
            # (BS, n_prop)
            top_proposals_class = torch.div(top_proposals, heatmap.shape[-1],
                                            rounding_mode='floor')
            # (BS, n_prop)
            top_proposals_index = top_proposals % heatmap.shape[-1]
            query_feat = lidar_feat_flatten.gather(
                index=top_proposals_index[:, None, :].expand(-1,
                                                             lidar_feat_flatten.shape[
                                                                 1], -1),
                dim=-1)
            # (BS, h_d, n_prop)
            self.query_labels = top_proposals_class  # (BS, n_prop)

            # add category embedding
            one_hot = F.one_hot(top_proposals_class,
                                num_classes=self.num_classes).permute(0, 2, 1)
            # [BS, n_prop, num_cls] -> [BS, num_cls, n_prop]
            query_cat_encoding = self.class_encoding(one_hot.float())
            # [BS, h_dim, n_prop]
            query_feat += query_cat_encoding  # (BS, h_d, n_prop)

            query_pos = bev_pos.gather(
                index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                    -1, -1, bev_pos.shape[-1]), dim=1)
            # [BS, n_prop, 2]
        else:
            query_feat = self.query_feat.repeat(batch_size, 1, 1)
            # [BS, h_d, n_prop]
            base_xyz = self.query_pos.repeat(batch_size, 1, 1).to(
                lidar_feats[0].device)  # [BS, n_prop, 2]

        # Transformer Decoder
        # lidar_feats is list[(bs, C, H, W), ... lvls]
        # img_feats is list[(bs * n_views, C, H, W), ... lvls]
        result_dicts = self.transformer(lidar_feats=lidar_feats,
                                        lidar_pos=bev_pos_mlvl_bs,
                                        img_feats=img_feats,
                                        img_pos=None,
                                        query=query_feat,
                                        query_pos=query_pos,
                                        pred_heads=self.prediction_heads,
                                        pred_heads_img=self.prediction_heads_img if self.with_img else None,
                                        img_metas=img_metas)

        if self.initialize_by_heatmap:
            result_dicts[0]['query_heatmap_score'] = heatmap.gather(
                index=top_proposals_index[:, None, :].expand(-1,
                                                             self.num_classes,
                                                             -1),
                dim=-1)  # (BS, n_cls, n_prop)
            if self.with_img:
                result_dicts[0]['dense_heatmap'] = dense_heatmap_img
            else:
                result_dicts[0]['dense_heatmap'] = dense_heatmap_lidar

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [result_dicts[-1]]

        # return all the layer's results for auxiliary supervision
        result_dict = {}
        for key in result_dicts[0].keys():
            if key not in ['dense_heatmap', 'dense_heatmap_old',
                           'query_heatmap_score']:
                result_dict[key] = torch.cat(
                    [ret_dict[key] for ret_dict in result_dicts], dim=-1)
                # [BS, chnl, n_dec * n_prop]
            else:
                result_dict[key] = result_dicts[0][key]
        return [result_dict]

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target. (BS, n_dec * n_prop)
                - torch.Tensor: classification weights (mask)
                    (BS, n_dec * n_prop)
                - torch.Tensor: regression target. (BS, n_dec * n_prop, 10)
                - torch.Tensor: regression weights. (BS, n_dec * n_prop, 10)
                - torch.Tensor: iou target. (BS, n_dec * n_prop)
                - int: number of positive proposals
                - float: mean iou
                - torch.Tensor: dense heatmap (BS, n_cls, H, W) H,W is 8 stride
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape (bs, 2, n_dec * n_prop)
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)
        # [dict('center':(1, 2, n_dec * n_prop), ...),. .. .BS]

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d,
                                gt_labels_3d, list_of_pred_dict,
                                np.arange(len(gt_labels_3d)))
        labels = torch.cat(res_tuple[0], dim=0)  # (BS, n_dec * n_prop)
        label_weights = torch.cat(res_tuple[1], dim=0)  # (BS, n_dec * n_prop)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        # (BS, n_dec * n_prop, 10)
        ious = torch.cat(res_tuple[4], dim=0)  # (BS, n_dec * n_prop)
        num_pos = np.sum(res_tuple[5])  # int
        matched_ious = np.mean(res_tuple[6])  # float
        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            # (BS, n_cls, H, W) H, W is 1440/8
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap
        else:
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict,
                           batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target. (1, n_dec * n_prop)
                - torch.Tensor: classification weights (mask) (1, n_dec * n_prop)
                - torch.Tensor: regression target. (1, n_dec * n_prop, 10)
                - torch.Tensor: regression weights. (1, n_dec * n_prop, 10)
                - torch.Tensor: iou target. (1, n_dec * n_prop)
                - int: number of positive proposals
                - float: mean iou
                - torch.Tensor: dense heatmap (1, n_cls, H, W) H,W is 8 stride
        """
        num_proposals = preds_dict['center'].shape[-1]  # (1, 2, n_dec*n_prop)

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        # [1, n_cls, n_dec * n_prop]
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height,
                                            vel)
        # decode the prediction to real world metric bbox
        # list of dict of samples, but there 1
        bboxes_tensor = boxes_dict[0]['bboxes']  # [n_dec * n_prop, 9]
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[
                                  self.num_proposals * idx_layer:self.num_proposals * (
                                              idx_layer + 1), :]
            # (n_prop, 9)
            score_layer = score[...,
                          self.num_proposals * idx_layer:self.num_proposals * (
                                      idx_layer + 1)]
            # (1, n_cls, n_prop)

            if self.train_cfg.assigner.type == 'HungarianAssignerDAFDeTr':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer,
                                                          gt_bboxes_tensor,
                                                          gt_labels_3d,
                                                          score_layer,
                                                          self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer,
                                                          gt_bboxes_tensor,
                                                          None, gt_labels_3d,
                                                          self.query_labels[
                                                              batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )  # (n_dec * n_prop, )
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   bboxes_tensor,
                                                   gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation  TRY n_dec * n_prop
        bbox_targets = torch.zeros(
            [num_proposals, self.bbox_coder.code_size]).to(
            center.device)
        bbox_weights = torch.zeros(
            [num_proposals, self.bbox_coder.code_size]).to(
            center.device)
        # (n_dec * n_prop, 10)

        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)

        labels = bboxes_tensor.new_zeros(num_proposals,
                                         dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals,
                                                dtype=torch.long)
        # (n_dec * n_prop, )

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat(
                [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
                dim=1).to(device)
            grid_size = torch.tensor(self.train_cfg['grid_size'])
            pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
            voxel_size = torch.tensor(self.train_cfg['voxel_size'])
            feature_map_size = torch.div(grid_size[:2], self.train_cfg[
                'out_size_factor'], rounding_mode='floor')
            # feature_map_size = grid_size[:2] // self.train_cfg[
            #     'out_size_factor']  # [x_len, y_len]
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes,
                                             feature_map_size[1],
                                             feature_map_size[0])
            # (n_cls, H, W) where H, W is 1440/8
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']
                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width),
                                             min_overlap=self.train_cfg[
                                                 'gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / \
                             self.train_cfg['out_size_factor']
                    coor_y = (y - pc_range[1]) / voxel_size[1] / \
                             self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                          center_int, radius)

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], \
                bbox_weights[None], ious[None], int(
                pos_inds.shape[0]), float(mean_iou), heatmap[None]
        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], \
                bbox_weights[None], ious[None], int(
                pos_inds.shape[0]), float(mean_iou)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for Mmf3DDeTr.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts ([list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap , class and bbox of each
                task.
        """
        if self.initialize_by_heatmap:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap = self.get_targets(
                gt_bboxes_3d, gt_labels_3d, preds_dicts)
            # # (BS, n_dec*n_prop), # (BS, n_dec*n_prop),
            # (BS, n_dec*n_prop, 10), (BS, n_dec*n_prop, 10)
            # (BS, n_dec*n_prop), int, float, # (BS, n_cls, H, W) HW is 1440/8
        else:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious = self.get_targets(
                gt_bboxes_3d, gt_labels_3d, preds_dicts)
        if hasattr(self, 'on_the_image_mask'):  # [BS, n_prop]
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            loss_heatmap = self.loss_heatmap(
                clip_sigmoid(preds_dict['dense_heatmap']), heatmap,
                avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(
                self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                    idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer * self.num_proposals:(
                                                                                  idx_layer + 1) * self.num_proposals].reshape(
                -1)
            layer_label_weights = label_weights[...,
                                  idx_layer * self.num_proposals:(
                                                                             idx_layer + 1) * self.num_proposals].reshape(
                -1)
            layer_score = preds_dict['heatmap'][...,
                          idx_layer * self.num_proposals:(
                                                                     idx_layer + 1) * self.num_proposals]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1,
                                                                   self.num_classes)
            # (BS * n_prop, n_cls)
            layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels,
                                           layer_label_weights,
                                           avg_factor=max(num_pos, 1))

            layer_center = preds_dict['center'][...,
                           idx_layer * self.num_proposals:(
                                                                      idx_layer + 1) * self.num_proposals]
            layer_height = preds_dict['height'][...,
                           idx_layer * self.num_proposals:(
                                                                      idx_layer + 1) * self.num_proposals]
            layer_rot = preds_dict['rot'][..., idx_layer * self.num_proposals:(
                                                                                          idx_layer + 1) * self.num_proposals]
            layer_dim = preds_dict['dim'][..., idx_layer * self.num_proposals:(
                                                                                          idx_layer + 1) * self.num_proposals]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)
            # (BS, n_prop, 8)
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][...,
                            idx_layer * self.num_proposals:(
                                                                       idx_layer + 1) * self.num_proposals]
                preds = torch.cat(
                    [layer_center, layer_height, layer_dim, layer_rot,
                     layer_vel], dim=1).permute(0, 2, 1)
                # (BS, num_proposals, 10)
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:,
                                 idx_layer * self.num_proposals:(
                                                                            idx_layer + 1) * self.num_proposals,
                                 :]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(
                code_weights)
            layer_bbox_targets = bbox_targets[:,
                                 idx_layer * self.num_proposals:(
                                                                            idx_layer + 1) * self.num_proposals,
                                 :]
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets,
                                             layer_reg_weights,
                                             avg_factor=max(num_pos, 1))

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f'matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False,
                   for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict['heatmap'].shape[0]
            batch_score = preds_dict['heatmap'][...,
                          -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            one_hot = F.one_hot(self.query_labels,
                                num_classes=self.num_classes).permute(0, 2, 1)
            # [BS, n_prop, n_cls] -> [BS, n_cls, n_prop]
            batch_score = batch_score * preds_dict[
                'query_heatmap_score'] * one_hot
            #  [BS, n_cls, n_prop]

            batch_center = preds_dict['center'][..., -self.num_proposals:]
            batch_height = preds_dict['height'][..., -self.num_proposals:]
            batch_dim = preds_dict['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim,
                                          batch_center, batch_height,
                                          batch_vel, filter=True)
            # list of dict with 'bboxes', 'scores' and 'labels'
            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[],
                         indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8],
                         radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'],
                         indices=[9], radius=0.175),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(num_class=1, class_names=['Car'], indices=[0],
                         radius=0.7),
                    dict(num_class=1, class_names=['Pedestrian'], indices=[1],
                         radius=0.7),
                    dict(num_class=1, class_names=['Cyclist'], indices=[2],
                         radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                # threshold score for tracking
                # mask = scores > 0.01
                # boxes3d = boxes3d[mask]
                # scores = scores[mask]
                # labels = labels[mask]
                ## adopt circle nms for different categories
                if self.test_cfg['nms_type'] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [boxes3d[task_mask][:, :2],
                                     scores[:, None][task_mask]], dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    img_metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg[
                                        'post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][
                                task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(bboxes=boxes3d[keep_mask],
                               scores=scores[keep_mask],
                               labels=labels[keep_mask])
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)  # [dict(), dict(), ... BS]
            rets.append(ret_layer)  # [[dict(), dict(), ... BS]]
        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [[
            img_metas[0]['box_type_3d'](rets[0][0]['bboxes'],
                                        box_dim=rets[0][0]['bboxes'].shape[
                                            -1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]
        return res

    def simple_test_bboxes(self, img_feats,
                           point_feats,
                           img_metas):
        """ Test det bboxes without test-time augmentation.
        Args:
            mlvl_feats (tuple[Tensor]): Multi0level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is an (n, 9) tensor, where the
                9 columns are bounding box positions
                (cx, cy, cz, l, w, h, theta, vx, vy). The second item is a (n,
                ) tensor where each item is predicted score between 0 and 1.
                The third item is a (n,) tensor where each item is the
                predicted class label of the corresponding box.
        """
        outs = self(img_feats, point_feats, img_metas)

        results_list = self.get_bboxes(outs, img_metas)
        return results_list


class PredHeads(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 **kwargs):
        super(PredHeads, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -center （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward
        Args:
            xyz (tensor): shape (BS, n_q, 2)
        """
        xyz = xyz.transpose(1, 2).contiguous()  # (BS, 2, n_q)
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding  # (BS, 128, n_q)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead,
                                                dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead,
                                                 dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(
                F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        query2 = \
            self.multihead_attn(
                query=self.with_pos_embed(query, query_pos_embed),
                key=self.with_pos_embed(key, key_pos_embed),
                value=self.with_pos_embed(key, key_pos_embed),
                attn_mask=attn_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(
            self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self,
                   '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,
                                 # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3,
                                                                          dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt,
                         in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt,
                         in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt,
                         in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)],
                                      dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask,
                     torch.zeros((key_padding_mask.size(0), 1),
                                 dtype=key_padding_mask.dtype,
                                 device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:],
                                      dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:],
                                      dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat(
                [attn_mask, torch.zeros((attn_mask.size(0), 1),
                                        dtype=attn_mask.dtype,
                                        device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)],
                dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len,
                                                src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len,
                                                       src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads,
                                                       tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p,
                                    training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz,
                                                                embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len,
                                                       src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
