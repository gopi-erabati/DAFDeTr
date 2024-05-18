plugin = True
plugin_dir = 'mmdet3d_plugin'

img_scale_ = 0.5
img_shape_ = [448, 768]  # (900, 1600) original
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8
# point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# sparse_shape = [41, 1440, 1440]
# grid_size = [1440, 1440, 40]
point_cloud_range = [-55.2, -55.2, -5.0, 55.2, 55.2, 3.0]
sparse_shape = [41, 1472, 1472]
grid_size = [1472, 1472, 40]

lidar_feat_lvls = 4
img_feat_lvls = 4

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='DAFDeTr',
    use_img=True,
    freeze_img=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        init_cfg=dict(type='Pretrained',
                      checkpoint='ckpts/lidar_cam_pretrain.pth',
                      prefix='img_backbone.'),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        norm_cfg=dict(type='BN2d'),
        relu_before_extra_convs=True,
        init_cfg=dict(type='Pretrained',
                      checkpoint='ckpts/lidar_cam_pretrain.pth',
                      prefix='img_neck.'),
    ),
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size, max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoderCustom',
        in_channels=5,
        sparse_shape=sparse_shape,
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
    ),
    pts_backbone=dict(
        type='SECONDCustom',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
    ),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[128, 256],
        out_channels=128,
        start_level=0,
        num_outs=4,
        add_extra_convs='on_output',
    ),
    bbox_head=dict(
        type='DAFDeTrHead',
        with_img=True,
        num_views=6,
        in_channels_img=256,
        lidar_feat_lvls=lidar_feat_lvls,
        img_feat_lvls=img_feat_lvls,
        num_proposals=300,
        auxiliary=True,
        in_channels_lidar=128,
        hidden_channel=128,
        num_classes=len(class_names),
        with_encoder=True,
        encoder_lidar=dict(
            type='DetrTransformerEncoder',
            num_layers=3,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=128, num_levels=lidar_feat_lvls),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=128,
                    feedforward_channels=256,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=256,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))
        ),
        # config for Transformer
        transformer=dict(
            type='DAFDeTrTransformer',
            decoder=dict(
                type='DAFDeTrTransformerDecoder',
                hidden_channel=128,
                num_layers=4,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=128,
                            num_heads=8,
                            dropout=0.1
                        ),
                        dict(
                            type='DAFDeTrCrossAtten',
                            embed_dims=128,
                            num_heads=8,
                            num_points=8,
                            lidar_feat_lvls=lidar_feat_lvls,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=128,
                        feedforward_channels=256,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm', 'cross_attn', 'norm',
                        'ffn', 'norm')
                ),
                num_layers_img=4,
                transformerlayers_img=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=128,
                            num_heads=8,
                            dropout=0.1
                        ),
                        dict(
                            type='DAFDeTrCrossAttenImage',
                            embed_dims=128,
                            img_feat_lvls=img_feat_lvls,
                            num_views=6,
                            out_size_factor_lidar=out_size_factor,
                            voxel_size=voxel_size[0],
                            pc_range_minx=point_cloud_range[0],
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=128,
                        feedforward_channels=256,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm', 'cross_attn', 'norm',
                        'ffn', 'norm')
                ),
                key_pos_emb_req=False,
                lidar_feat_lvls=lidar_feat_lvls,
            )
        ),
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        num_heads=8,
        ffn_channel=256,  # in TF
        dropout=0.1,  # in TF
        bn_momentum=0.1,
        activation='relu',
        # config for FFN
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2),
                          vel=(2, 2)),
        bbox_coder=dict(
            type='DAFDeTrBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25,
                      reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean',
                          loss_weight=1.0),
        init_cfg=None,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssignerDAFDeTr',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25,
                              weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range,
        )
    ),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        ),
    ),
)

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[img_scale_],
         img_shape=img_shape_),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='HorizontalRandomFlipMultiViewImage', flip_ratio=0.5),
    # dict(type='RandomFlip3DMultiViewImage', flip_ratio_bev_horizontal=0.5),
    # Flip should be after RandomScaleImageMultiViewImage
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img',
                                 'points'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[img_scale_],
         img_shape=img_shape_),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(type='HorizontalRandomFlipMultiViewImage', flip_ratio=0.5),
            dict(type='PointsRangeFilter',
                 point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img', 'points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'img'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
# for 8gpu * 2sample_per_gpu
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 10
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None

# update the path of LiDAR-only model below to load the weights for fusion
# model
load_from = 'ckpts/dafdetr_res_voxel_L.pth'

resume_from = None
workflow = [('train', 1)]

# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=1, pipeline=eval_pipeline)
freeze_lidar_components = True
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
find_unused_parameters = True
