## Pytorch -> ONNX -> NART
NART工具安装文档[url](http://compile-link.pages.gitlab.bj.sensetime.com/nart/index.html)

## Requirements
尽量使用本地虚拟环境进行转换
* nart >= 1.2.9dev
* 3.7 >= python >= 3.6
* torch=1.3.1
* thop

## cli
### convert to tensorrt backend
```shell script
cd tools
bash to_nart_local.sh ../experiments/GG18 tensorrt
```
### convert to onnx for ADELA
```shell script
cd tools
bash to_nart_local.sh ../experiments/GG18 onnx
```
you must copy content of trt_nart_cfg.yml | nart_config_for_adela.yml to adela website for different backend convert
### convert z_decoder.onnx to integer version
```shell script
cd tools/nart
python convert_to_integer_onnx.py -path ../q0.2.7_4/z_decoder.onnx
```
### count Params and Flops via to_nart_local.sh shell
```shell script
cd tools
bash to_nart_local.sh ../experiments/GG18 flops
```