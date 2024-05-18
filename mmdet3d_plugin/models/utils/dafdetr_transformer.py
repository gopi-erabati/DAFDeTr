import copy
import warnings
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network,
                                         build_norm_layer,
                                         build_transformer_layer)
from mmcv.runner import force_fp32
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils import ConfigDict
from mmdet.models.utils.builder import TRANSFORMER


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


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class DAFDeTrTransformer(BaseModule):
    """
    Implements the DAFDeTr Transformer
    """

    def __init__(self, decoder):
        super(DAFDeTrTransformer, self).__init__()
        self.decoder = build_transformer_layer_sequence(decoder)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DAFDeTrCrossAtten):
                m.init_weights()

    def forward(self,
                lidar_feats,
                lidar_pos,
                img_feats,
                img_pos,
                query,
                query_pos,
                pred_heads=None,
                pred_heads_img=None,
                img_metas=None,
                ):
        """ Forward function of TF
        Args:
            lidar_feats (list[Tensor]): List of LiDAR feats of shape (BS,
                h_dim, H, W)
            lidar_pos (list[Tensor]): List of LiDAR feats position encoding of
                shape (BS, H*W, 2)
            img_feats (list[Tensor]): Image feats list of shape
                (BS*N, h_dim, H, W) or None
            img_pos (Tensor): Image feats pos or None
            query (Tensor): Query feats of shape (BS, h_dim, n_prop)
            query_pos (Tensor): Query position of shape (BS, n_prop, h_dim)
            pred_heads (nn.ModuleList()): Prediction heads for decoder layers
            pred_heads_img (nn.ModuleList()): Pred heads for Image in
                cross-attention when using with image feats
        Returns:
            result_dicts (list[dict]): A list of result dicts for each layer
        """

        result_dicts = self.decoder(query,
                                    query_pos,
                                    lidar_feats,
                                    lidar_pos,
                                    img_feats,
                                    img_pos,
                                    pred_heads,
                                    pred_heads_img,
                                    img_metas
                                    )

        return result_dicts


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DAFDeTrTransformerDecoder(TransformerLayerSequence):
    """ Decoder Layers
    Args:
        transformerlayers (dict): Config for Transformer layers
        num_layers (int): number of decoder layers
    """

    def __init__(self, transformerlayers=None, num_layers=None,
                 transformerlayers_img=None, num_layers_img=None,
                 hidden_channel=128, key_pos_emb_req=False, lidar_feat_lvls=1):
        # LiDAR
        super(DAFDeTrTransformerDecoder, self).__init__(
            transformerlayers=transformerlayers, num_layers=num_layers)
        self.query_pos_emb = nn.ModuleList()
        for _ in range(len(self.layers)):
            self.query_pos_emb.append(
                PositionEmbeddingLearned(2, hidden_channel))

        # IMAGE
        self.num_layers_img = num_layers_img
        if self.num_layers_img is not None:
            if isinstance(transformerlayers_img, dict):
                transformerlayers_img = [
                    copy.deepcopy(transformerlayers_img) for _ in range(
                        num_layers_img)
                ]
            else:
                assert isinstance(transformerlayers_img, list) and \
                       len(transformerlayers_img) == num_layers_img
            self.layers_img = ModuleList()
            for i in range(num_layers_img):
                self.layers_img.append(build_transformer_layer(
                    transformerlayers_img[i]))
            self.query_pos_emb_img = nn.ModuleList()
            for _ in range(len(self.layers_img)):
                self.query_pos_emb_img.append(
                    PositionEmbeddingLearned(2, hidden_channel))

        # This for MHA (not used though)
        self.key_pos_emb_req = key_pos_emb_req
        if key_pos_emb_req:  # req if using MHA
            self.key_pos_emb = nn.ModuleList()
            for _ in range(lidar_feat_lvls):
                self.key_pos_emb.append(
                    PositionEmbeddingLearned(2, hidden_channel))

    def forward(self,
                query,
                query_pos,
                lidar_feats,
                lidar_pos,
                img_feats,
                img_pos,
                pred_heads=None,
                pred_heads_img=None,
                img_metas=None):
        """ Forward of Decoder
        Args:
            query (Tensor): Query feats of shape (BS, h_dim, n_prop)
            query_pos (Tensor): Query position of shape (BS, n_prop, 2)
            lidar_feats (list[Tensor]): List of LiDAR feats of shape (BS,
                h_dim, H, W)
            lidar_pos (list[Tensor]): List of LiDAR feats position encoding of
                shape (BS, H*W, 2)
            img_feats (list[Tensor]): Image feats list of shape
                (BS*N, h_dim, H, W) or None
            img_pos (Tensor): Image feats pos or None
            pred_heads (nn.ModuleList()): Prediction heads for decoder
                layers for LiDAR
            pred_heads_img (nn.ModuleList()): Pred heads for Image in
                cross-attention when using with image feats
        Returns:
            result_dicts (list[dict]): A list of result dicts for each layer
        """

        # put batch last for query
        query = query.permute(2, 0, 1)  # (n_prop, BS, h_dim)

        result_dicts = []
        output = query

        batch_size = lidar_feats[0].shape[0]

        # LiDAR
        for lid, layer in enumerate(self.layers):
            # get query position embeddings
            query_pos_emb = self.query_pos_emb[lid](query_pos)
            # (BS, h_dim, n_prop)
            query_pos_emb = query_pos_emb.permute(2, 0, 1)
            # (n_prop, BS, h_dim)

            # This is for MHA (not used!)
            # get key position embeddings
            key_pos_emb_list = []
            if self.key_pos_emb_req:
                for lvl, lidar_pos_lvl in enumerate(lidar_pos):
                    key_pos_emb = self.key_pos_emb[lvl](lidar_pos_lvl)
                    # (BS, h_dim, H*W)
                    key_pos_emb = key_pos_emb.permute(2, 0, 1)
                    # (H*W, BS, h_dim)
                    key_pos_emb_list.append(key_pos_emb)
            lidar_feat_flatten = lidar_feats[0].flatten(2).permute(2, 0, 1)
            # (H*W, BS, h_dim)

            # Goes to BaseTransformerLayer to perform
            # self-attn, norm, cross-attn, norm, ffn , norm
            output = layer(output,
                           key=lidar_feats,
                           value=lidar_feats,
                           query_pos=query_pos_emb,
                           query_pos_org=query_pos,
                           key_pos=key_pos_emb_list[0] if self.key_pos_emb_req
                           else None,
                           img_metas=img_metas)
            # (n_prop, BS, h_dim)

            # Prediction
            output = output.permute(1, 2, 0)
            # (n_prop, BS, h_dim) -> (BS, h_dim, n_prop)
            result_layer = pred_heads[lid](output)
            output = output.permute(2, 0, 1)
            # (BS, h_dim, n_prop) -> (n_prop, BS, h_dim)
            result_layer['center'] = result_layer['center'] + \
                                     query_pos.permute(0, 2, 1)
            # (BS, 2, n_prop)
            # for next level positional embedding
            query_pos = result_layer['center'].detach().clone().permute(0, 2,
                                                                        1)
            # (BS, n_prop, 2)

            if self.num_layers_img is None:
                result_dicts.append(result_layer)

        # IMAGE
        if self.num_layers_img is not None:

            output_lidar = output.clone()  # LiDAR query feat to concatenate
            # (n_prop, BS, h_dim)
            output_lidar = output_lidar.permute(1, 2, 0)
            # (n_prop, BS, h_dim) -> (BS, h_dim, n_prop)

            for lid, layer in enumerate(self.layers_img):
                # get query position embeddings
                query_pos_emb_img = self.query_pos_emb_img[lid](query_pos)
                # (BS, h_dim, n_prop)
                query_pos_emb_img = query_pos_emb_img.permute(2, 0, 1)
                # (n_prop, BS, h_dim)

                # This is for MHA (not used!)
                # get key position embeddings
                key_pos_emb_list = []
                if self.key_pos_emb_req:
                    for lvl, lidar_pos_lvl in enumerate(lidar_pos):
                        key_pos_emb = self.key_pos_emb[lvl](lidar_pos_lvl)
                        # (BS, h_dim, H*W)
                        key_pos_emb = key_pos_emb.permute(2, 0, 1)
                        # (H*W, BS, h_dim)
                        key_pos_emb_list.append(key_pos_emb)
                lidar_feat_flatten = lidar_feats[0].flatten(2).permute(2, 0, 1)
                # (H*W, BS, h_dim)

                # Goes to BaseTransformerLayer to perform
                # self-attn, norm, cross-attn, norm, ffn , norm
                output = layer(output,
                               key=img_feats,
                               value=img_feats,
                               query_pos=query_pos_emb_img,
                               query_pos_org=query_pos,
                               key_pos=key_pos_emb_list[
                                   0] if self.key_pos_emb_req
                               else None,
                               query_pos_height=result_layer['height'],
                               batch_size=batch_size,
                               img_metas=img_metas)
                # (n_prop, BS, h_dim)

                # Prediction
                output = output.permute(1, 2, 0)
                # (n_prop, BS, h_dim) -> (BS, h_dim, n_prop)
                output_fused = torch.cat([output_lidar, output], dim=1)
                # (BS, 2*h_dim, n_prop)
                result_layer = pred_heads_img[lid](output_fused)
                output = output.permute(2, 0, 1)
                # (BS, h_dim, n_prop) -> (n_prop, BS, h_dim)
                result_layer['center'] = result_layer['center'] + \
                                         query_pos.permute(0, 2, 1)
                # (BS, 2, n_prop)
                # for next level positional embedding
                query_pos = result_layer['center'].detach().clone().permute(0,
                                                                            2,
                                                                            1)
                # (BS, n_prop, 2)

                result_dicts.append(result_layer)

        return result_dicts  # list[dict]


@ATTENTION.register_module()
class DAFDeTrCrossAtten(BaseModule):
    """ Cross-Attention to perform attention on LiDAR feat maps
    It uses MultiScaleDeformableCrossAttention to perform the attention

    Args:

    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 num_points=4,
                 lidar_feat_lvls=1,
                 init_cfg=None,
                 batch_first=False
                 ):
        super(DAFDeTrCrossAtten, self).__init__(init_cfg=init_cfg)

        # multi-scale cross-atten for LiDAR
        self.msdca_lidar = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=lidar_feat_lvls,
            num_points=num_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                query_pos_org=None,
                key_pos=None,
                **kwargs
                ):
        """ Forward
        Args:
            query (Tensor): Query of shape (n_prop, BS, h_dim)
            key (List[Tensor, list[Tensor]]): A list of LiDAR and Image feats.
                The first item is LiDAR feats of shape (BS, h_dim, H, W)
                and the second item is Image feats of list of different
                levels, of shape (BS*N, h_dim, H, W)
            value (List[Tensor, list[Tensor]]): Same as Key
            query_pos (Tensor): Query pos of shape (n_prop, BS, h_dim)
            query_pos_org (Tensor): Query pos original values (BS, n_prop, 2)
            key_pos (List[Tensor, list[Tensor]]): A list of LiDAR and Image
                feats positions. The first is LiDAR BEV pos of shape
                (BS, H*W, 2) and
        """
        feats_lidar = key
        # feats_lidar is list[(bs, C, H, W), ... lvls]

        # LiDAR
        feats_lidar_spatial_shape_mlvl = []
        feat_lidar_mlvl = []
        for idx, feat_lidar in enumerate(feats_lidar):
            bs, dim_lidar, h_lidar, w_lidar = feat_lidar.shape
            feat_lidar_spatial_shape = (h_lidar, w_lidar)
            feats_lidar_spatial_shape_mlvl.append(feat_lidar_spatial_shape)
            feat_lidar = feat_lidar.flatten(2).permute(0, 2, 1)
            # (BS, H*W, h_dim)
            feat_lidar_mlvl.append(feat_lidar)
        # concatenate all levels
        feat_lidar_mlvl = torch.cat(feat_lidar_mlvl, dim=1)
        # (BS, lvl*H*W, h_dim)
        feat_lidar_mlvl = feat_lidar_mlvl.permute(1, 0, 2)
        # (lvl*H*W, BS, h_dim)

        spatial_shape_lidar = torch.as_tensor(feats_lidar_spatial_shape_mlvl,
                                              dtype=torch.long,
                                              device=feat_lidar.device)

        reference_points_lidar = query_pos_org.float()  # (BS, n_prop, 2)
        reference_points_lidar = reference_points_lidar.unsqueeze(2).repeat(
            1, 1, len(feats_lidar), 1)
        # (BS, n_prop, lvls, 2)
        reference_points_lidar[..., 0] /= spatial_shape_lidar[0, 0]
        reference_points_lidar[..., 1] /= spatial_shape_lidar[0, 1]
        # normalize the ref points

        lvl_start_index_lidar = torch.cat(
            (spatial_shape_lidar.new_zeros(
                (1,)),
             spatial_shape_lidar.prod(1).cumsum(0)[:-1]))  # (lvls, )

        # query is (n_prop, BS, h_dim)
        # feat_lidar_mlvl is (lvl*H*W, BS, h_dim)
        # query_pos is (n_prop, BS, h_dim)
        # reference_points_lidar is (BS, n_prop, lvls, 2)
        # spatial_shape_lidar is (lvls, 2)
        # lvl_start_index_lidar is (lvls, )
        query = self.msdca_lidar(query,
                                 key=feat_lidar_mlvl,
                                 value=feat_lidar_mlvl,
                                 query_pos=query_pos,
                                 reference_points=reference_points_lidar,
                                 spatial_shapes=spatial_shape_lidar,
                                 level_start_index=lvl_start_index_lidar)
        #  (n_prop, BS, h_dim)

        return query


@ATTENTION.register_module()
class DAFDeTrCrossAttenImage(BaseModule):
    """ Cross-Attention to perform attention on Camera feat maps
    It uses MultiScaleDeformableCrossAttention to perform the attention

    Args:

    """

    def __init__(self,
                 embed_dims,
                 img_feat_lvls=4,
                 num_views=0,
                 out_size_factor_lidar=8,
                 voxel_size=0.075,
                 pc_range_minx=None,
                 init_cfg=None,
                 batch_first=False
                 ):
        super(DAFDeTrCrossAttenImage, self).__init__(init_cfg=init_cfg)

        # Camera parameters
        self.num_views = num_views
        if not self.num_views == 0:
            self.level_embeds = nn.Parameter(torch.Tensor(
                img_feat_lvls, embed_dims))
            self.cams_embeds = nn.Parameter(
                torch.Tensor(num_views, embed_dims))

            # layer to get query_pos embeddings
            self.query_pos_emb = PositionEmbeddingLearned(2, embed_dims)

            # parameters to get real 3D query pos
            self.out_size_factor_lidar = out_size_factor_lidar
            self.voxel_size = voxel_size
            self.pc_range_minx = pc_range_minx

            # attention
            self.msdca_image = build_attention(
                dict(
                    type='ImageCrossAttention',
                    embed_dims=embed_dims,
                    num_cams=num_views,
                    attn_cfg=dict(
                        type='MSDeformableAttentionImage',
                        embed_dims=embed_dims,
                        num_levels=img_feat_lvls
                    )
                )
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or \
                    isinstance(m, ImageCrossAttention) or \
                    isinstance(m, MSDeformableAttentionImage):
                m.init_weights()
        if not self.num_views == 0:
            normal_(self.level_embeds)
            normal_(self.cams_embeds)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                query_pos_org=None,
                query_pos_height=None,
                batch_size=None,
                img_metas=None,
                key_pos=None,
                **kwargs
                ):
        """ Forward
        Args:
            query (Tensor): Query of shape (n_prop, BS, h_dim)
            key (List[Tensor, list[Tensor]]): A list of LiDAR and Image feats.
                The first item is LiDAR feats of shape (BS, h_dim, H, W)
                and the second item is Image feats of list of different
                levels, of shape (BS*N, h_dim, H, W)
            value (List[Tensor, list[Tensor]]): Same as Key
            query_pos (Tensor): Query pos of shape (n_prop, BS, h_dim)
            query_pos_org (Tensor): Query pos original values (BS, n_prop, 2)
            key_pos (List[Tensor, list[Tensor]]): A list of LiDAR and Image
                feats positions. The first is LiDAR BEV pos of shape
                (BS, H*W, 2) and
        """
        feats_img = key
        # feats_img is list[(bs * n_views, C, H, W), ... lvls]

        query_pos_realmetric = query_pos_org.permute(0, 2, 1) * \
                               self.out_size_factor_lidar * \
                               self.voxel_size + self.pc_range_minx
        # (BS, 2, n_prop)
        query_pos_3d = torch.cat(
            [query_pos_realmetric, query_pos_height],
            dim=1).detach().clone()
        # (BS, 3, n_prop)
        query_pos_3d = query_pos_3d.permute(0, 2, 1)  # (BS, n_prop, 3)

        # get ref pts in 2d and mask for n_views
        ref_pts_cam, query_mask = self.get_ref2d_mask(query_pos_3d,
                                                      img_metas)
        # (n_cam, BS, n_prop, 2) (n_cam, BS, n_prop)

        spatial_shapes_img = []
        feat_img_flatten = []
        for lvl, feat_img in enumerate(feats_img):
            # feat_img of shape (bs * n_views, h_dim, H, W)
            img_h, img_w, num_channel = feat_img.shape[-2], \
                feat_img.shape[-1], \
                feat_img.shape[1]
            feat_img = feat_img.view(batch_size, self.num_views,
                                     num_channel, img_h,
                                     img_w)
            # (BS, n_views, h_dim, H, W)
            spatial_shape = (img_h, img_w)
            feat_img = feat_img.flatten(3).permute(1, 0, 3, 2)
            # (n_views, BS, H*W, h_dim)
            feat_img = feat_img + self.cams_embeds[:, None, None, :].to(
                feat_img.dtype)
            feat_img = feat_img + self.level_embeds[None,
                                  None, lvl:lvl + 1, :].to(feat_img.dtype)
            spatial_shapes_img.append(spatial_shape)
            feat_img_flatten.append(feat_img)

        feat_img_flatten = torch.cat(feat_img_flatten, 2)
        # # (n_views, BS, lvls*H*W, h_dim)
        spatial_shapes_img = torch.as_tensor(
            spatial_shapes_img, dtype=torch.long, device=feat_img.device)
        level_start_index_img = torch.cat((spatial_shapes_img.new_zeros(
            (1,)), spatial_shapes_img.prod(1).cumsum(0)[:-1]))

        feat_img_flatten = feat_img_flatten.permute(0, 2, 1, 3)
        # (n_views, lvls*H*W, BS, h_dim)

        # attention
        query = query.permute(1, 0, 2)
        # (n_prop, BS, h_dim) -> (BS, n_prop, h_dim)
        query_pos = query_pos.permute(1, 0, 2)
        # (n_prop, BS, h_dim) -> (BS, n_prop, h_dim)

        # query is (BS, n_prop, h_dim)
        # feat_img_flatten is (n_views, lvls*H*W, BS, h_dim)
        # query_pos_emb is (BS, n_prop, h_dim)
        # ref_pts_cam is (n_cam, BS, n_prop, 2)
        # query_mask is (n_cam, BS, n_prop)
        query = self.msdca_image(query=query, key=feat_img_flatten,
                                 value=feat_img_flatten,
                                 query_pos=query_pos,
                                 spatial_shapes=spatial_shapes_img,
                                 reference_points_cam=ref_pts_cam,
                                 query_mask=query_mask,
                                 level_start_index=level_start_index_img,
                                 )
        # (BS, n_prop, h_dim)

        # permute query back to original shape
        query = query.permute(1, 0, 2)
        # (BS, n_prop, h_dim) -> (n_prop, BS, h_dim)

        return query

    def get_ref2d_mask(self, ref_3d, img_metas):
        """ Function to get reference points projected onto 2D images along
        with mask
        Args:
            ref_3d (Tensor): Reference points in 3D of shape (BS, n_prop, 3)
            img_metas (list[dict]): img_metas for all images
        """
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = ref_3d.new_tensor(lidar2img)  # (B, N, 4, 4)

        ref_3d = torch.cat((ref_3d, torch.ones_like(ref_3d[..., :1])), -1)
        # (BS, n_prop, 4)

        B, n_prop = ref_3d.size()[:2]
        num_cam = lidar2img.size(1)

        ref_3d = ref_3d.view(B, 1, n_prop, 4).repeat(1, num_cam, 1, 1
                                                     ).unsqueeze(-1)
        # (BS, n_cam, n_prop, 4, 1)

        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, n_prop, 1,
                                                               1)
        # (BS, n_cam, n_prop, 4, 4)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            ref_3d.to(torch.float32)).squeeze(
            -1)
        # (BS, n_cam, n_prop, 4)
        eps = 1e-5

        query_mask = (reference_points_cam[..., 2:3] > eps)
        # (BS, n_cam, n_prop, 1)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        # (BS, n_cam, n_prop, 2)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        query_mask = (query_mask & (reference_points_cam[..., 1:2] > 0.0)
                      & (reference_points_cam[..., 1:2] < 1.0)
                      & (reference_points_cam[..., 0:1] < 1.0)
                      & (reference_points_cam[..., 0:1] > 0.0))
        query_mask = torch.nan_to_num(query_mask)

        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3)
        # (BS, n_cam, n_prop, 2) -> (n_cam, BS, n_prop, 2)
        query_mask = query_mask.permute(1, 0, 2, 3).squeeze(-1)
        # (BS, n_cam, n_prop, 1) -> (n_cam, BS, n_prop)

        return reference_points_cam, query_mask
        # (n_cam, BS, n_prop, 2) (n_cam, BS, n_prop)


@ATTENTION.register_module()
class ImageCrossAttention(BaseModule):
    """ This class calculates cross-attention of query with Image features"""

    def __init__(self,
                 embed_dims=128,
                 num_cams=6,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 attn_cfg=dict(
                     type='MSDeformableAttentionImage',
                     embed_dims=128,
                     num_levels=4),
                 **kwargs
                 ):
        super(ImageCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False
        self.deformable_attention = build_attention(attn_cfg)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=(
            'query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                reference_points_cam=None,
                query_mask=None,
                level_start_index=None,
                **kwargs
                ):
        """ Forward fun
        Args:
            query (Tensor): shape (BS, n_q, d)
            reference_points_cam (Tensor): shape (n_cam, BS, n_q, 2)
            key (Tensor): shape (n_cam, lvls*H*W, BS, h_dim)
            value (Tensor): shape (n_cam, lvls*H*W, BS, h_dim)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)  # (bs, n_q, d)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        indexes_per_cam = []
        max_len_all = []
        for mask_per_cam in query_mask:  # (n_cam, BS, n_prop)
            indexes_per_img = []
            for mask_per_img in mask_per_cam:
                index_query_per_img = mask_per_img.nonzero().squeeze(-1)
                indexes_per_img.append(index_query_per_img)
            indexes_per_cam.append(indexes_per_img)
            max_len_per_cam = max([len(each) for each in indexes_per_img])
            max_len_all.append(max_len_per_cam)
        max_len = max(max_len_all)
        # index_query_per_img provides indices of nonzero queries of shape (
        # n_q, ) for each cam and each batch ; max_len provides maximum of
        # query indices for all cams and all batches
        # indexes_per_img is a list of indices for all batches
        # indexes_per_cam is a list of above list

        # each camera only interacts with its corresponding BEV queries.
        # This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        # (bs, n_cam, max_len, d)
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, 2])
        # (bs, n_cam, max_len, 2)

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes_per_cam[i][j]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[
                    j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = \
                    reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)  # (BS * n_cam, lvl*H*W, d)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)  # (BS * n_cam, lvl*H*W, d)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len,
                                       self.embed_dims), key=key, value=value,
            reference_points=reference_points_rebatch.view(bs * self.num_cams,
                                                           max_len, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index).view(bs, self.num_cams,
                                                      max_len, self.embed_dims)
        # (BS, max_len, h_dim) -> (BS, n_cams, max_len, h_dim)

        for j in range(bs):
            for i, index_cam in enumerate(indexes_per_cam):
                index_query_per_img = index_cam[j]
                slots[j, index_query_per_img] += queries[j, i,
                                                 :len(index_query_per_img)]
        # (BS, n_prop, h_dim)

        count = query_mask > 0  # (n_cam, BS, n_prop)
        count = count.permute(1, 2, 0).sum(-1)  # (BS, n_prop)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]  # (BS, n_prop, 1)
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttentionImage(BaseModule):
    """ Implements Deformable Attention for Multi-view Images
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            reference_points = reference_points.unsqueeze(2).repeat(1, 1,
                                                                    self.num_levels,
                                                                    1)
            # (BS, n_q, n_lvls, 2)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets \
                                 / offset_normalizer[None, None, None, :, None,
                                   :]
            # (BS, n_q, n_h, n_lvls, n_points, 2)
        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output  # (BS, n_q, d)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DAFDeTrTransformerDecoderOld(TransformerLayerSequence):
    """ Decoder Layers
    Args:
        transformerlayers (dict): Config for Transformer layers
        num_layers (int): number of decoder layers
    """

    def __init__(self, transformerlayers=None, num_layers=None,
                 hidden_channel=128, key_pos_emb_req=False, lidar_feat_lvls=1):
        super(DAFDeTrTransformerDecoderOld, self).__init__(
            transformerlayers=transformerlayers, num_layers=num_layers)
        self.query_pos_emb = nn.ModuleList()
        for _ in range(len(self.layers)):
            self.query_pos_emb.append(PositionEmbeddingLearned(2,
                                                               hidden_channel))
        self.key_pos_emb_req = key_pos_emb_req
        if key_pos_emb_req:  # req if using MHA
            self.key_pos_emb = nn.ModuleList()
            for _ in range(lidar_feat_lvls):
                self.key_pos_emb.append(PositionEmbeddingLearned(2,
                                                                 hidden_channel))

    def forward(self,
                query,
                query_pos,
                lidar_feats,
                lidar_pos,
                img_feats,
                img_pos,
                pred_heads=None,
                pred_heads_lidar=None,
                img_metas=None):
        """ Forward of Decoder
        Args:
            query (Tensor): Query feats of shape (BS, h_dim, n_prop)
            query_pos (Tensor): Query position of shape (BS, n_prop, 2)
            lidar_feats (list[Tensor]): List of LiDAR feats of shape (BS,
                h_dim, H, W)
            lidar_pos (list[Tensor]): List of LiDAR feats position encoding of
                shape (BS, H*W, 2)
            img_feats (list[Tensor]): Image feats list of shape
                (BS*N, h_dim, H, W) or None
            img_pos (Tensor): Image feats pos or None
            pred_heads (nn.ModuleList()): Prediction heads for decoder layers
            pred_heads_lidar (nn.ModuleList()): Pred heads for LiDAR in
                cross-attention when using with image feats
        Returns:
            result_dicts (list[dict]): A list of result dicts for each layer
        """

        # put batch last for query
        query = query.permute(2, 0, 1)  # (n_prop, BS, h_dim)

        result_dicts = []
        output = query

        for lid, layer in enumerate(self.layers):
            # get query position embeddings
            query_pos_emb = self.query_pos_emb[lid](query_pos)
            # (BS, h_dim, n_prop)
            query_pos_emb = query_pos_emb.permute(2, 0, 1)
            # (n_prop, BS, h_dim)

            # get key position embeddings
            key_pos_emb_list = []
            if self.key_pos_emb_req:
                for lvl, lidar_pos_lvl in enumerate(lidar_pos):
                    key_pos_emb = self.key_pos_emb[lvl](lidar_pos_lvl)
                    # (BS, h_dim, H*W)
                    key_pos_emb = key_pos_emb.permute(2, 0, 1)
                    # (H*W, BS, h_dim)
                    key_pos_emb_list.append(key_pos_emb)
            lidar_feat_flatten = lidar_feats[0].flatten(2).permute(2, 0, 1)
            # (H*W, BS, h_dim)

            # Goes to BaseTransformerLayer to perform
            # self-attn, norm, cross-attn, norm, ffn , norm
            output = layer(output,
                           key=[lidar_feats, img_feats],  # lidar_feat_flatten,
                           value=[lidar_feats, img_feats],
                           # lidar_feat_flatten
                           query_pos=query_pos_emb,
                           query_pos_org=query_pos,
                           key_pos=key_pos_emb_list[0] if self.key_pos_emb_req
                           else None,
                           pred_heads_lidar=pred_heads_lidar[lid] if
                           pred_heads_lidar is not None else None,
                           img_metas=img_metas)
            # (n_prop, BS, h_dim)

            # for img + LiDAR fusion output is a tuple of query and query_pos
            if img_feats is not None:
                output, query_pos = output
                # (n_prop, BS, h_dim), (BS, n_prop, 2)

            # Prediction
            output = output.permute(1, 2, 0)
            # (n_prop, BS, h_dim) -> (BS, h_dim, n_prop)
            result_layer = pred_heads[lid](output)
            output = output.permute(2, 0, 1)
            # (BS, h_dim, n_prop) -> (n_prop, BS, h_dim)
            result_layer['center'] = result_layer['center'] + \
                                     query_pos.permute(0, 2, 1)
            # (BS, 2, n_prop)

            result_dicts.append(result_layer)

            # for next level positional embedding
            query_pos = result_layer['center'].detach().clone().permute(0, 2,
                                                                        1)
            # (BS, n_prop, 2)

        return result_dicts  # list[dict]


@ATTENTION.register_module()
class DAFDeTrCrossAttenOld(BaseModule):
    """ Cross-Attention to perform attention on LiDAR and Camera feat maps
    It uses MultiScaleDeformableCrossAttention to perform the attention

    Args:

    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 num_points=4,
                 lidar_feat_lvls=1,
                 img_feat_lvls=4,
                 num_views=0,
                 fuse_with_se=True,
                 out_size_factor_lidar=8,
                 voxel_size=0.075,
                 pc_range_minx=None,
                 init_cfg=None,
                 batch_first=False
                 ):
        super(DAFDeTrCrossAttenOld, self).__init__(init_cfg=init_cfg)

        # multi-scale cross-atten for LiDAR
        self.msdca_lidar = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=lidar_feat_lvls,
            num_points=num_points)

        # Camera parameters
        self.num_views = num_views
        if not self.num_views == 0:
            self.level_embeds = nn.Parameter(torch.Tensor(
                img_feat_lvls, embed_dims))
            self.cams_embeds = nn.Parameter(
                torch.Tensor(num_views, embed_dims))

            # layer to get query_pos embeddings
            self.query_pos_emb = PositionEmbeddingLearned(2, embed_dims)

            # parameters to get real 3D query pos
            self.out_size_factor_lidar = out_size_factor_lidar
            self.voxel_size = voxel_size
            self.pc_range_minx = pc_range_minx

            # attention
            self.msdca_image = build_attention(
                dict(
                    type='ImageCrossAttention',
                    embed_dims=embed_dims,
                    num_cams=num_views,
                    attn_cfg=dict(
                        type='MSDeformableAttentionImage',
                        embed_dims=embed_dims,
                        num_levels=img_feat_lvls
                    )
                )
            )

            self.feat_fusion = nn.Conv1d(in_channels=embed_dims * 2,
                                         out_channels=embed_dims,
                                         kernel_size=1)

            self.fuse_with_se = fuse_with_se
            if self.fuse_with_se:
                self.fuse_with_se_conv = nn.Conv1d(in_channels=embed_dims,
                                                   out_channels=embed_dims,
                                                   kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or \
                    isinstance(m, ImageCrossAttention) or \
                    isinstance(m, MSDeformableAttentionImage):
                m.init_weights()
        if not self.num_views == 0:
            normal_(self.level_embeds)
            normal_(self.cams_embeds)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                query_pos_org=None,
                pred_heads_lidar=None,
                img_metas=None,
                key_pos=None,
                **kwargs
                ):
        """ Forward
        Args:
            query (Tensor): Query of shape (n_prop, BS, h_dim)
            key (List[Tensor, list[Tensor]]): A list of LiDAR and Image feats.
                The first item is LiDAR feats of shape (BS, h_dim, H, W)
                and the second item is Image feats of list of different
                levels, of shape (BS*N, h_dim, H, W)
            value (List[Tensor, list[Tensor]]): Same as Key
            query_pos (Tensor): Query pos of shape (n_prop, BS, h_dim)
            query_pos_org (Tensor): Query pos original values (BS, n_prop, 2)
            key_pos (List[Tensor, list[Tensor]]): A list of LiDAR and Image
                feats positions. The first is LiDAR BEV pos of shape
                (BS, H*W, 2) and
        """
        feats_lidar, feats_img = key
        # feats_lidar is list[(bs, C, H, W), ... lvls]
        # feats_img is list[(bs * n_views, C, H, W), ... lvls]

        # LiDAR
        batch_size = feats_lidar[0].shape[0]
        feats_lidar_spatial_shape_mlvl = []
        feat_lidar_mlvl = []
        for idx, feat_lidar in enumerate(feats_lidar):
            bs, dim_lidar, h_lidar, w_lidar = feat_lidar.shape
            feat_lidar_spatial_shape = (h_lidar, w_lidar)
            feats_lidar_spatial_shape_mlvl.append(feat_lidar_spatial_shape)
            feat_lidar = feat_lidar.flatten(2).permute(0, 2, 1)
            # (BS, H*W, h_dim)
            feat_lidar_mlvl.append(feat_lidar)
        # concatenate all levels
        feat_lidar_mlvl = torch.cat(feat_lidar_mlvl, dim=1)
        # (BS, lvl*H*W, h_dim)
        feat_lidar_mlvl = feat_lidar_mlvl.permute(1, 0, 2)
        # (lvl*H*W, BS, h_dim)

        spatial_shape_lidar = torch.as_tensor(feats_lidar_spatial_shape_mlvl,
                                              dtype=torch.long,
                                              device=feat_lidar.device)

        reference_points_lidar = query_pos_org.float()  # (BS, n_prop, 2)
        reference_points_lidar = reference_points_lidar.unsqueeze(2).repeat(
            1, 1, len(feats_lidar), 1)
        # (BS, n_prop, lvls, 2)
        reference_points_lidar[..., 0] /= spatial_shape_lidar[0, 0]
        reference_points_lidar[..., 1] /= spatial_shape_lidar[0, 1]
        # normalize the ref points

        lvl_start_index_lidar = torch.cat(
            (spatial_shape_lidar.new_zeros(
                (1,)),
             spatial_shape_lidar.prod(1).cumsum(0)[:-1]))  # (lvls, )

        # query is (n_prop, BS, h_dim)
        # feat_lidar_mlvl is (lvl*H*W, BS, h_dim)
        # query_pos is (n_prop, BS, h_dim)
        # reference_points_lidar is (BS, n_prop, lvls, 2)
        # spatial_shape_lidar is (lvls, 2)
        # lvl_start_index_lidar is (lvls, )
        query = self.msdca_lidar(query,
                                 key=feat_lidar_mlvl,
                                 value=feat_lidar_mlvl,
                                 query_pos=query_pos,
                                 reference_points=reference_points_lidar,
                                 spatial_shapes=spatial_shape_lidar,
                                 level_start_index=lvl_start_index_lidar)
        #  (n_prop, BS, h_dim)

        if feats_img is not None:
            # get the LiDAR predictions
            query = query.permute(1, 2, 0)
            # (n_prop, BS, h_dim) -> (BS, h_dim, n_prop)
            result_layer_lidar = pred_heads_lidar(query)
            query = query.permute(2, 0, 1)
            # (BS, h_dim, n_prop) -> (n_prop, BS, h_dim)
            result_layer_lidar['center'] = result_layer_lidar['center'] + \
                                           query_pos_org.permute(0, 2, 1)
            # (BS, 2, n_prop)

            # for next image cross-attn
            query_lidar = query.detach().clone()  # for concat with query_img
            query_pos = result_layer_lidar['center'].detach().clone().permute(
                0, 2, 1)  # (BS, n_prop, 2)
            query_pos_emb = self.query_pos_emb(query_pos)
            # (BS, h_dim, n_prop)
            query_pos_emb = query_pos_emb.permute(2, 0, 1)
            # (n_prop, BS, h_dim)
            query_pos_realmetric = query_pos.permute(0, 2, 1) * \
                                   self.out_size_factor_lidar * \
                                   self.voxel_size + self.pc_range_minx
            query_pos_3d = torch.cat(
                [query_pos_realmetric, result_layer_lidar['height']],
                dim=1).detach().clone()
            # (BS, 3, n_prop)
            query_pos_3d = query_pos_3d.permute(0, 2, 1)  # (BS, n_prop, 3)

            # get ref pts in 2d and mask for n_views
            ref_pts_cam, query_mask = self.get_ref2d_mask(query_pos_3d,
                                                          img_metas)
            # (n_cam, BS, n_prop, 2) (n_cam, BS, n_prop)

            spatial_shapes_img = []
            feat_img_flatten = []
            for lvl, feat_img in enumerate(feats_img):
                # feat_img of shape (bs * n_views, h_dim, H, W)
                img_h, img_w, num_channel = feat_img.shape[-2], \
                    feat_img.shape[-1], \
                    feat_img.shape[1]
                feat_img = feat_img.view(batch_size, self.num_views,
                                         num_channel, img_h,
                                         img_w)
                # (BS, n_views, h_dim, H, W)
                spatial_shape = (img_h, img_w)
                feat_img = feat_img.flatten(3).permute(1, 0, 3, 2)
                # (n_views, BS, H*W, h_dim)
                feat_img = feat_img + self.cams_embeds[:, None, None, :].to(
                    feat_img.dtype)
                feat_img = feat_img + self.level_embeds[None,
                                      None, lvl:lvl + 1, :].to(feat_img.dtype)
                spatial_shapes_img.append(spatial_shape)
                feat_img_flatten.append(feat_img)

            feat_img_flatten = torch.cat(feat_img_flatten, 2)
            # # (n_views, BS, lvls*H*W, h_dim)
            spatial_shapes_img = torch.as_tensor(
                spatial_shapes_img, dtype=torch.long, device=feat_img.device)
            level_start_index_img = torch.cat((spatial_shapes_img.new_zeros(
                (1,)), spatial_shapes_img.prod(1).cumsum(0)[:-1]))

            feat_img_flatten = feat_img_flatten.permute(0, 2, 1, 3)
            # (n_views, lvls*H*W, BS, h_dim)

            # attention
            query_lidar = query_lidar.permute(1, 0, 2)
            # (n_prop, BS, h_dim) -> (BS, n_prop, h_dim)
            query_pos_emb = query_pos_emb.permute(1, 0, 2)
            # (n_prop, BS, h_dim) -> (BS, n_prop, h_dim)

            # query is (BS, n_prop, h_dim)
            # feat_img_flatten is (n_views, lvls*H*W, BS, h_dim)
            # query_pos_emb is (BS, n_prop, h_dim)
            # ref_pts_cam is (n_cam, BS, n_prop, 2)
            # query_mask is (n_cam, BS, n_prop)
            query_img = self.msdca_image(query=query_lidar,
                                         key=feat_img_flatten,
                                         value=feat_img_flatten,
                                         query_pos=query_pos_emb,
                                         spatial_shapes=spatial_shapes_img,
                                         reference_points_cam=ref_pts_cam,
                                         query_mask=query_mask,
                                         level_start_index=level_start_index_img,
                                         )
            # (BS, n_prop, h_dim)

            # reshape query and query_img to (BS, h_dim, n_prop)
            query_lidar = query_lidar.permute(0, 2, 1)  # (BS, h_dim, n_prop)
            query_img = query_img.permute(0, 2, 1)  # (BS, h_dim, n_prop)

            # Fusion of LiDAR and Image Queries
            query_fused = torch.cat([query_lidar, query_img], dim=1)
            # (BS, h_dim * 2, n_prop)
            query_fused = self.feat_fusion(query_fused)
            # (BS, h_dim, n_prop)

            if self.fuse_with_se:
                query_fused_attn = self.fuse_with_se_conv(query_fused)
                # (BS, h_dim, n_prop)
                query_fused_attn = query_fused_attn.sigmoid_()
                # (BS, h_dim, n_prop)
                query_fused = query_fused_attn * query_fused

            return query_fused.permute(2, 0, 1), query_pos
            # (n_prop, BS, h_dim), (BS, n_prop, 2)
        else:
            return query

    def get_ref2d_mask(self, ref_3d, img_metas):
        """ Function to get reference points projected onto 2D images along
        with mask
        Args:
            ref_3d (Tensor): Reference points in 3D of shape (BS, n_prop, 3)
            img_metas (list[dict]): img_metas for all images
        """
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = ref_3d.new_tensor(lidar2img)  # (B, N, 4, 4)

        ref_3d = torch.cat((ref_3d, torch.ones_like(ref_3d[..., :1])), -1)
        # (BS, n_prop, 4)

        B, n_prop = ref_3d.size()[:2]
        num_cam = lidar2img.size(1)

        ref_3d = ref_3d.view(B, 1, n_prop, 4).repeat(1, num_cam, 1, 1
                                                     ).unsqueeze(-1)
        # (BS, n_cam, n_prop, 4, 1)

        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, n_prop, 1,
                                                               1)
        # (BS, n_cam, n_prop, 4, 4)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            ref_3d.to(torch.float32)).squeeze(
            -1)
        # (BS, n_cam, n_prop, 4)
        eps = 1e-5

        query_mask = (reference_points_cam[..., 2:3] > eps)
        # (BS, n_cam, n_prop, 1)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        # (BS, n_cam, n_prop, 2)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        query_mask = (query_mask & (reference_points_cam[..., 1:2] > 0.0)
                      & (reference_points_cam[..., 1:2] < 1.0)
                      & (reference_points_cam[..., 0:1] < 1.0)
                      & (reference_points_cam[..., 0:1] > 0.0))
        query_mask = torch.nan_to_num(query_mask)

        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3)
        # (BS, n_cam, n_prop, 2) -> (n_cam, BS, n_prop, 2)
        query_mask = query_mask.permute(1, 0, 2, 3).squeeze(-1)
        # (BS, n_cam, n_prop, 1) -> (n_cam, BS, n_prop)

        return reference_points_cam, query_mask
        # (n_cam, BS, n_prop, 2) (n_cam, BS, n_prop)


# Custom transformer_layer and base_transformer to include tuple output from
# cross-attention block
class BaseTransformerLayerCustom(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
               set(operation_order), f'The operation_order of' \
                                     f' {self.__class__.__name__} should ' \
                                     f'contains all four operation type ' \
                                     f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                                               f'of attn_cfg {num_attn} is ' \
                                               f'not consistent with the number of attention' \
                                               f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                # check if img_feats is not None
                # because cross-attn outputs tuple when its not None
                if value[1] is not None:
                    query, querypos = query
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        if value[1] is not None:
            return query, querypos
        else:
            return query


@TRANSFORMER_LAYER.register_module()
class DAFDeTrTransformerDecoderLayer(BaseTransformerLayerCustom):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DAFDeTrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
