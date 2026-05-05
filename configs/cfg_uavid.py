_base_ = './base_config.py'



# model settings
model = dict(    
    name_path='./configs/cls_uavid.txt',
    prob_thd=0.3,
    slide_stride=224,
    slide_crop=224
)

# dataset settings
dataset_type = 'UAVidDataset'
data_root = './datasets/UAVid/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/0_1080_1280_2560', seg_map_path='ann_dir/0_1080_1280_2560'),
        pipeline=test_pipeline))