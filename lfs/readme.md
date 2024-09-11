# 概述
建议先简单的阅读confluence上spring1关于dispatcher和sci接口的文档，理解关于这套流程的概念，如sci, task, stage。大量模板代码的实现细节不必须关注。

目前是通过dispatcher运行，程序实际入口是该项目目录下的__main__.py。stage文件是lfs_stage，里面包含了训练流程信息。主要调用的其它文件还有lfs文件夹下的。

loss_func_search中包含了控制搜索的基本函数。lfs_agent仅仅负责采样。loss中提供了不同的搜索空间。reward中则是不同的reward。

采样和REINFORCE算法本身的实现不复杂，主要是论文eq19-20，ats release的代码中在计算一次reward内部进行了多次的采样和计算logit，因此与常见的eq20实现不一致。

另外实现中会涉及分布式训练的一些问题。

# 实现细节
 - lfs_stage中通过simple_group_split进行分组
 - _common_build本身有设置seed作用，同时dataset的sample中又设了一次，最后在它后面我们再设置一次。只用看最后一次
 - 向task_config（即playground中的config一样）注入lfs键值之后，会在初始化时传入task_helper构造。结果是会传到_data_pool里。arg/lfs键值下。在LossProcess中会再次取出去get_loss。
 - dataset中sampler有修改。提供了多卡测试保持顺序、固定dataset采样顺序的基础。

# evaluate现状
verbose提供单卡测所有图功能。dataset为config中的test。
fast提供多卡测所有图功能。dataset为config中的fast_test。除此之外还有区分测试集和验证集作用。

# auto_train
```
python auto_train.py
```

base目录下是基本的配置。

# 一些名称
 - a就是指采样后的结果
 - mu指采样均值
 - scale指采样方差

# 配置解读
例如，某base目录中pipeline.yaml会引入task_config gg18.yaml和stage_config codec_stage.yaml

所有lfs有关配置都在codec_stage.yaml中
 - start_sample_iter是开始sample的迭代数
 - sample_step，sample采样一次参数，之后的loss用这次的采样结果求，但是不会update
 - lfs_start_update_iter是开始update的迭代数
 - update_freq，update会计算reward，之后就是automl那些步骤。必须是sample_step倍数，不然上一个mu的sample会沿用到这个周期里。
 - lfs_loc是a的初始值
 - reward_func，loss_func都是动态注册机制。
 - 其它都是想到了就代码里一条线加进去[捂脸]。

# lfs_agent
通过继承agent中提供的基类，实现了不同具体的采样方式。包括
 - 将采样结果clip到0，1
 - 将采样结果clip到0，inf
 - etc.

# loss
loss_func_search通过传递参数a提供搜索系数，通过传参lambda_dis提供distortion权重

# 其他机制
reward_func的动态注册实现在spring_helper/agility_task_helper.py，这是codec代码的sci接口，会被dispatcher调用。对应的配置位置在gg18.yaml中的type: AgilityTaskHelper。

disaptcher里面的lfs_stage.py由’LFSStage‘配置，在pipeline.yaml里，负责控制整个训练流程。


