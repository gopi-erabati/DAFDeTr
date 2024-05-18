plugin = True
plugin_dir = 'mmdet3d_plugin'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.1, 0.1, 0.15]
out_size_factor = 8
# point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# sparse_shape = [41, 1440, 1440]
# grid_size = [1440, 1440, 40]
# point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]
point_cloud_range = [-76.8, -76.8, -2, 76.8, 76.8, 4]
sparse_shape = [41, 1536, 1536]
grid_size = [1536, 1536, 40]

lidar_feat_lvls = 4

class_names = ['Car', 'Pedestrian', 'Cyclist']

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='DAFDeTrWaymo',
    use_img=False,
    pts_voxel_layer=dict(
        max_num_points=5,
        voxel_size=voxel_size,
        max_voxels=150000,
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=5,
        # num_features=5,
        feat_channels=[64],
        with_distance=False,
        with_cluster_center=False,
        with_voxel_center=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoderCustom',
        in_channels=64,
        sparse_shape=sparse_shape,
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=(
        (16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
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
        with_img=False,
        lidar_feat_lvls=lidar_feat_lvls,
        num_proposals=300,
        auxiliary=True,
        in_channels_lidar=128,
        hidden_channel=128,
        num_classes=len(class_names),
        with_encoder=True,
        encoder_lidar=dict(
            type='DetrTransformerEncoder',
            num_layers=2,
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
                num_layers=4,
                hidden_channel=128,
                key_pos_emb_req=False,
                lidar_feat_lvls=lidar_feat_lvls,
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
                )
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
                          ),
        bbox_coder=dict(
            type='DAFDeTrBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0],
            score_threshold=0.0,
            code_size=8,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25,
                      reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2.0),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean',
                          loss_weight=1.0),
        init_cfg=None,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            dataset='Waymo',
            assigner=dict(
                type='HungarianAssignerDAFDeTr',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25,
                              weight=0.6),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=2.0),
                iou_cost=dict(type='IoU3DCost', weight=2.0)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=grid_size,  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=point_cloud_range
        )
    ),
    test_cfg=dict(
        pts=dict(
            dataset='Waymo',
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            nms_type=None,
        )
    ),
)

dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample',
         db_sampler=dict(
             data_root=data_root,
             info_path=data_root + '/waymo_dbinfos_train.pkl',
             rate=1.0,
             prepare=dict(
                 filter_by_difficulty=[-1],
                 filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
             classes=class_names,
             sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
             points_loader=dict(
                 type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6,
                 use_dim=[0, 1, 2, 3, 4]))
         ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(800, 1333),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
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
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            load_interval=1,
            ann_file=data_root + '/waymo_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/waymo_infos_val.pkl',
        split='training',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/waymo_infos_val.pkl',
        split='training',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 4sample_per_gpu
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
total_epochs = 36
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(interval=36, pipeline=eval_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

find_unused_parameters = True
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'