# run the test in agility_task_helper
```shell script
cd ..
srun --mpi=pmi2 -p VI_AIC_TITANXP -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --job-name=test python -m spring_helper.agility_task_helper  | tee log
```
# some change in config
```yaml
# after 25 eval epoch we save image for one time
show_img_interval: 25  

# when use train stage we use default eval that is
# as_val: True
# func_mode: verbose

eval:  # use in codec eval stage 
    as_val: False
    func_mode: verbose

eval_oneshot:  # use in codec nas stage
    as_val: True
    func_mode: fast

eval_in_train:  # use in show log and only effective when specify show_bpp_interval
    as_val: True
    func_mode: fast

metrics:  
    msssim: 1.

nas: False

dataset: 
  fast_test:  # must
    meta_file_list:
      - /mnt/lustre/share/fe/codec_list/trainset.txt 
  data_augment:
    train:
      - img_random_crop:
          crop_size: 256
    test: ~
    fast_test:
      - img_random_crop:
          crop_size: 256

```
# auto train
```
cd spring_helper
mkdir your_auto_train_path
cp -r your_base_config_dir your_auto_train_path
cp -r auto_train/auto_train.py your_auto_train_path 
cd your_auto_train_path
vim auto_train.py 
```