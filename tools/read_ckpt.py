import torch



path = "./results/patchcore/mvtec/bottle-cls/weights/model.ckpt"


# 包含训练参数
l_model = torch.load(path)
print(l_model.keys())
# dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks',
# 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters', 'hparams_type'])


# 模型参数
#print(l_model['state_dict'].keys())
# dict_keys(['image_threshold.value', 'pixel_threshold.value', 'training_distribution.image_mean',
# 'training_distribution.image_std', 'training_distribution.pixel_mean', 'training_distribution.pixel_std',
# 'min_max.min', 'min_max.max', 'model.memory_bank', 'model.feature_extractor.backbone.conv1.weight',其余是模型训练好的权重...])