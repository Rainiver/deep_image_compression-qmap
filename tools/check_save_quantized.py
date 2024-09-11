import torch

ckpt = torch.load('../experiments/q/logs/codec_epoch-0.pth', map_location='cpu')
print(ckpt.keys())
print(ckpt['module.sub_models.z_decoder.deconv1.weight'].abs().min())
print(ckpt['module.sub_models.z_decoder.deconv1.fw'].abs().min())
print(ckpt['module.sub_models.z_decoder.deconv1.bias'].abs().min())
print(ckpt['module.sub_models.z_decoder.deconv1.fb'].abs().min())
print(ckpt['module.sub_models.z_decoder.act1.c'][0].abs().min())

ckpt = torch.load('../experiments/q/logs/codec_epoch-1.pth', map_location='cpu')
print(ckpt['module.sub_models.z_decoder.deconv1.weight'].abs().min())
print(ckpt['module.sub_models.z_decoder.deconv1.fw'].abs().min())
print(ckpt['module.sub_models.z_decoder.deconv1.bias'].abs().min())
print(ckpt['module.sub_models.z_decoder.deconv1.fb'].abs().min())
print(ckpt['module.sub_models.z_decoder.act1.c'][0].abs().min())
