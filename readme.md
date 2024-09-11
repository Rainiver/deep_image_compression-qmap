# introduction

Image compression based on DL.

使用过程中发现任何问题请及时修改本文档，运行的python文件应该是读过的，黑箱操作不可取

# painless first try

在`/codes/cc`里make一下，编译熵编码部分

```shell
source s0.3.2
cd codes/cc
make
cd ../..
```

在`/jpeg/jpeg_decoder`里make一下，编译jpeg解码工具(可执行程序和.so库)，同时运行脚本生成jpeg量化表

```bash
source s0.3.2
cd jpeg/jpeg_decoder
make
python get_QT.py
cd ../..
```

安装相关pip依赖

```shell
source s0.3.2
pip install --upgrade pip --user  # upgrade pip
pip install tensorboard --user  # used for display training process
pip install -r requirements.txt --user  # install pip dependences
```

(可选)安装torchjpeg，DCT相关实验需要

see: https://gitlab.bj.sensetime.com/facedet/codec/torchjpeg

编译wheel并将torchjpeg源码置于项目根目录下

(可选)安装apex加速训练
```shell script
# clone and active environment
conda create -n s0.3.2-apex --clone r0.3.2
source activate s0.3.2-apex

# clone source
git clone https://github.com/NVIDIA/apex
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0

# mannuly set torch.utils.cpp_extension.CUDA_HOME to correct one
# add this:
#
# [setup.py]
# from torch.utils import cpp_extension
# cpp_extension.CUDA_HOME = '/mnt/lustre/share/cuda-9.0'
vim setup.py

# prepare
mkdir $HOME/tmp
export TMPDIR=$HOME/tmp
conda install -c psi4 gcc-5

# install
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ --user

# remove gcc-5
conda uninstall gcc-5
```

开始训练：

```shell  
source s0.3.2
export PYTHONPATH=.:$PYTHONPATH

cd tools
bash train.sh spring_scheduler ../experiments/GG18/
```
跑整条RD曲线的所有模型，请参考auto_train和auto_test部分

开始量化训练：  
注意：使用V1（integer2）的训练，val时没有更新到test_cfg，必须运行test_quant.sh才能得到正确精度  
```shell   
git submodule update --init  
bash train_quant.sh ../experiments/1dn_GG18_quantV1/ ../experiments/integer_configs/warm_w8_a8.yaml
```

查看训练状况（基于tensorboard）：

```shell
tensorboard --logdir experiments --host 0.0.0.0 --port 16384
```

可以在`16384`端口查看当前训练状况（在集群内也可以查看，可以直接使用集群ip访问指定端口）

spring.submit 手册：
https://confluence.sensetime.com/pages/viewpage.action?pageId=242925857

# auto_train
  tools/auto_train.py，根据输入的模型和权重类型来进行自动训练（一般只需要输入-tp和-pi即可）
  自动训练结束后，生成的模型在'../experiments/目录下，名称为auto_Train_' + 选择的权重类型拼接而成
1. '-tp','--loss_weight_parameter_type',选择权重类型，有psnr,msssim,hybrid,grad,grad_mse,0.2.3,cheng20mse,gg18mse,all192mse共9种类型
2. '-pi', '--input_dir_name',输入模型路径
3. '-dp', '--use_default_cfg_dir',default=True,help='default cfg dir prefix ../experiments else none'
4. '-re', '--restore_dir',default='../experiments,'help='default exp dir ../experiments else input'
5. '--range', type=str, default='*',help='range of exp index, a python expression or "*"'
6. '--partition', '-P', type=str, default='VI_AIC_TITANXP',选择分区

# auto test
   tools/auto_test.py
1. 自动合并同一个数据集（目录）下所有单点模型的结果   
    merge_val_tee_log = True   
    运行 python auto_test.py   
    现在是合并/mnt/lustre/share/hedailan/gg18-ms-curve中模型的最后一次validation结果，gg18-ssim-11-B被排除在外。         
    另一个例子，合并auto train的val结果：    

2. 拿这些模型批量测试所有数据  
    mode = 'test'   
    merge_val_tee_log = False  
    运行  
3. 批测完批量合并结果并保存csv到result   
    mode = 'merge'   
    merge_val_tee_log = False  
    运行  
    这一步如果遇到没有正常测试结果的实验会给出提示，删掉有问题实验的tee_test.log返回2重新测试即可  
    不会重复测试没有问题的实验。另外测试有问题也可能是炸显存等原因  

其余可能需要修改的项： 
overwrite=False，如果result中有同名csv文件则跳过不测试   
partition，save_path_pattern，max_testing，models_dir，models_exclude  

# to caffe

拉nart代码：

```shell
git submodule update --init   
cd nart/python   
python setup.py install
cd tools   
bash to_caffe.sh VI_AIC_TITANXP ../exp_dir
```

* submodule的使用方法：https://blog.csdn.net/weixin_42995876/article/details/86255625

统计参数量和flops:

如果是本地转换，注意按照集群环境配置py36, torch和torchvision　
```shell
bash to_caffe_local.sh ../exp_dir   
```

统计参数量和flops:  
```shell  
python -m spring.nart.tools.caffe.count caffe/y_decoder.prototxt
```

# caffe to nart/trt
用自己的switch库需要art权限来编译　　   
要求cuda10, trt7，在本地转（或集群的docker，但是trt模型要求显卡架构和cuda trt版本一致，集群转完不方便测试　　　
要求python 3.6   
bash to_nart.sh  

# compare codec
激活动态链接库环境

```shell
source /mnt/lustre/share/fe/wangliangzhou/codec/heif_env.sh
```

下面这些依赖已经全放在share, test_codec应该可以直接运行, 不需要再下载和配置这些库.  

test_webp
需要安装libwebp: https://developers.google.com/speed/webp/docs/precompiled?hl=zh-CN
下载编译好的即可  

```shell
export PATH=$PATH:../libwebp-1.0.3-linux-x86-64/bin/
```

test_bpg (as tf paper)  https://bellard.org/bpg/ 

jpeg: opencv  

jpeg2000: https://jpeg.org/jpeg2000/software.html, 选择其中的OpenJPEG as tf paper  https://github.com/uclouvain/openjpeg 

openjpeg, libwebp和libbpg放了一份在/mnt/lustre/share/fe/wangyan1/codec

# train

## parameters:

tools/train.py 中修改参数：

`log_dir`: 模型参数路径

`data_dir`:训练数据路径 (`/mnt/lustre/share/zhengyaoyan/dataset/compression` 34集群)

`load_epoch`:指定加载哪一套参数(来自`log_dir`)

## run:

### CPU:

```shell
source s0.3.2
export PYTHONPATH=.:$PYTHONPATH
srun -p VI_AIC_1080TI python tools/train.py
```

### GPU:

train.sh:

```shell
work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n16 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=1 \
python -u train.py
```

shell:

```shell
./train.sh VI_AIC_1080TI
```



# test

## parameters:

tools/test.py:

`main(img_path,base,log_dir,epoch)`: 单图测试，输出到当前目录下`output.png`,压缩后文件输出到当前目录下`test.jpg`

	image_path:测试图片路径
	
	base:需要pad原图边长到base的整数倍
	
	log_dir:加载参数路径
	
	epoch:指定加载哪一套参数

`test(data_dir)`:测试一个文件夹内所有图，输出平均ssim, psnr

	data_dir:测试图片路径(`/mnt/lustre/share/zhengyaoyan/dataset/compression/valid` 34集群)

## run:

### CPU:

```shell
python test.py
```

### GPU:

test.sh:

```shell
work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n16 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=1 \
python -u test.py
```

shell:

```shell
./test.sh VI_AIC_1080TI
```

### GPU测速
开启`--test_time`选项，使每个process运行后执行一次cuda同步，确保GPU测速准确性

shell:

```shell
./test_cuda_sync.sh spring_scheduler <EXPERIMENT_ROOT>
```
### tensorboard
```shell
tensorboard --logdir=logs --port 19988
```

### unit test
```shell
# install pytest
pip install pytest

# run all test cases
py.test test
```

# Pipeline and New Entry

**(building)**

新的入口代码为`tools/playground.py`, 目前运行参数与方法与`train_val`一致

功能:
- train
- test
- compress
- decompress (WIP)
- to_caffe (WIP)

## 兼容性与迁移

- 新代码把所有网络参数放置在一个参数文件`codec_epoch-xxx.pth`中，老代码训练出的参数文件需要把所有子网络的参数放在一个名为`models`的`state_dict`
- 老的配置文件需要添加一个`codec`(暂定名)选项来选择/定义使用的pipeline.
  - 对于标准gg17, 添加`codec: gg17`
  - 对于标准gg18, 添加`codec: gg18`
  - 对于其他pipeline暂无内置支持，可以按照下文中Pipeline配置的方法添加配置定义
  
## dynamic model builder

当配置中存在`model_arch`字段时，将动态加载并定义网络结构。**注意此时原本的模型加载参数将全部失效**

可以在`vars`字段中定义全局变量

配置例：

```yaml
model_arch:   # required
    y_encoder:
        import: nets.decoder.decoders  # required
        args:  # optional
            - YGG17  # tag
        kwargs:  # optional
            out_channels: $COMMON_NUM_CHANNELS
            in_channels: $COMMON_NUM_CHANNELS
    model_name2:
        ...

vars:  # optional, variable definitions
    COMMON_NUM_CHANNELS: 256
    var2: ...
    var3: ...
```

## Pipeline定义与使用

### 添加新的Process类代码

继承`pipelines.processes.BaseProcess`或其子类并实现`run`接口来定义新的`Process`

`run`方法中：
  - 访问`self._data_pool[name]`属性读写数据池中名为`name`的数据
  - 访问`self._models[name]`属性获取模型池中名为`name`的子模型

原则上`run`方法的side-effect只有对数据池内少部分数据的更新 

### 添加新的Pipeline类代码

继承`pipelines.models.BaseCodec`, 并定义其`process_cls_list`属性

`process_cls_list`应当为一个`list`对象，其每个元素都是一个`BaseProcess`子类或一个`dict`对象，描述需要顺序执行的每个Process

当其中一个元素是`dict`对象时，这个`dict`将添加一个Process并说明它会在什么时候被Pipeline执行。它应该满足如下约束:
- 必须包含键`process`，其对应的值为一个`BaseProcess`的子类
- 可选键`only`，其对应值为一个`CodecStageEnum`枚举或由`CodecStageEnum`枚举组成的`set`对象。表示只有在这些stage当前Process才会被执行。若缺省则默认为全部stage组成的全集
- 可选键`exclude`，其对应值为一个`CodecStageEnum`枚举或由`CodecStageEnum`枚举组成的`set`对象。表示在这些stage时当前Process一定不会被执行

注意，当一个process的only-stage集合和exclude-stage集合交集非空时，Pipeline不会在处于交集内的stage时执行当前Process。
即：实际会执行当前Process的集合`set(exec) = set(only) - set(exclude)`

**不需要在`process_cls_list`中定义有关Loss计算和除了BPP以外的验证指标计算**

例子：参考`pipelines.models`下已有的Pipeline定义即可

### 在配置文件中动态定义Pipeline

借助YAML配置文件实现了较为初级的DSL。如果需要测试并频繁修改新的Pipeline，可以在配置文件的`coden`选项中进行定义。

以Contextual的DSL配置定义为例说明：

```yaml
codec:
   processes:
     - pipelines.processes.YEncodeProcess          # 根据路径动态加载Process所在模块
     - pipelines.processes.YQuantizeProcess
     - pipelines.processes.YDecodeProcess
     - pipelines.processes.ZAbsEncodeProcess
     - pipelines.processes.ZQuantizeProcess
     - pipelines.processes.ZDecodeProcess
     - pipelines.processes.ContextProcess
     - pipelines.processes.PriorAndContextParameterProcess
 

     - pipelines.processes.ZFactorizedEntropyProcess
     - pipelines.processes.GMMEntropyProcess
 
     # 嵌套一个数组对象来定义exclude和only
     - 
       process: pipelines.processes.YAlignSigmaProcess
       exclude: TRAIN    # exclude和only的含义与代码中一致；使用stage名称表示stage
 
     - 
       process: pipelines.processes.YGaussianModelCompressProcess
       exclude: TRAIN
     - 
       process: pipelines.processes.ZFactorizedModelCompressProcess
       exclude: TRAIN

     - 
       process: pipelines.processes.YZRealBPPEvalProcess
       only:    # 使用列表对象表示stage集合
         - VALID
         - TEST
```

当需要定义尚不存在代码实现的Process时，可以借助`attrs`字段向Process模板中注入属性动态定义。

以只使用Hyper Prior而不使用Context的Pipeline为例：

```yaml
codec:
   processes:
     - pipelines.processes.YEncodeProcess
     - pipelines.processes.YQuantizeProcess
     - pipelines.processes.YDecodeProcess
     - pipelines.processes.ZAbsEncodeProcess
     - pipelines.processes.ZQuantizeProcess
     - pipelines.processes.ZDecodeProcess
 
     # 使用一个LambdaProcess生成0张量
     - 
       process: pipelines.processes.BaseLambdaProcess
       attrs:
         input_name: prior
         output_name: tmp/zeros    # 添加了'tmp/'前缀的中间数据不会被tensorboard记录
         lambda_str: 'lambda x: torch.zeros_like(x).cuda()'  # 符合python语法的lambda表达式

     # 使用生成的0张量替代context
     - 
       process: pipelines.processes.BaseMapProcess
       attrs:
         input_name: tmp/zeros
         output_name: y_context

     - pipelines.processes.PriorAndContextParameterProcess

     - pipelines.processes.ZFactorizedEntropyProcess
     - pipelines.processes.GMMEntropyProcess
 
     - 
       process: pipelines.processes.YAlignSigmaProcess
       exclude: TRAIN
 
     - 
       process: pipelines.processes.YGaussianModelCompressProcess
       exclude: TRAIN
     - 
       process: pipelines.processes.ZFactorizedModelCompressProcess
       exclude: TRAIN
 
     - 
       process: pipelines.processes.YZRealBPPEvalProcess
       only:
         - VALID
         - TEST
```

### 数据键名约定

- `loss/`前缀的数据全部表示需要汇总到tensorboard的loss值。其中`loss/total`是加权求和的总和，用来计算BP的梯度
- `eval/`前缀的数据全部表示验证指标，应为numpy数组格式
- `image/`前缀的数据表示需要被保存/记录的图片张量
- `tmp/`前缀的数据表示被tensorboard忽略的中间值
- `arg/`前缀的数据为传入Pipeline的参数
- 其余数据名，无论是否有前缀，均会被tensorboard跟踪记录

此外，名字以`_likelihoods`结尾的任何变量将被视为概率并用以计算熵损失(码长)

# result

![](http://chuantu.xyz/t6/702/1568723181x3661913030.png)![](http://chuantu.xyz/t6/702/1568723222x989499252.png)