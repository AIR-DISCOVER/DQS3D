_base_ = ['semi_fcaf3d.py']
n_points = 100000

model = dict(
    model_cfg=dict(
        neck_with_head=dict(
            n_classes=10,
            n_reg_outs=8)),
    transformation=dict(
        # flipping=True,
        rotation_angle="orthogonal",
        # rotation_angle=None,
        translation_offset=0.5,  # 0.5 meters
        # scaling_factor=0.00
    ),
    semi_loss_parameters=dict(
        thres_center=0.4,
        thres_cls=0.2,
    ),
    # disable_QEC=True,
    eval_teacher=True,
    alpha=0.999,
    semi_loss_weights=dict(
        weight_consistency_bboxes = 1.00,
        weight_consistency_center = 0.25,
        weight_consistency_cls = 0.50,
    ),)

dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='IndoorPointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(type='IndoorPointSample', num_points=n_points),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
            type='SemiDataset',
            labeled=dict(
                type='LabeledDataset',
                seed=5,
                src=dict(
                    type=dataset_type,
                    modality=dict(use_camera=False, use_lidar=True),
                    data_root=data_root,
                    ann_file=data_root + 'sunrgbd_infos_train.pkl',
                    pipeline=train_pipeline,
                    filter_empty_gt=True,
                    classes=class_names,
                    box_type_3d='Depth'),
                ratio=0.05
            ),
            unlabeled=dict(
                type='UnlabeledDataset',
                seed=5,
                src=dict(
                    type=dataset_type,
                    modality=dict(use_camera=False, use_lidar=True),
                    data_root=data_root,
                    ann_file=data_root + 'sunrgbd_infos_train.pkl',
                    pipeline=train_pipeline,
                    filter_empty_gt=True,
                    classes=class_names,
                    box_type_3d='Depth'),
                ratio=0.00
            )),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

find_unused_parameters=True