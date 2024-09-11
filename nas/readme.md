## Helper bout the Directory Tree
```shell script
mkdir nas
cd nas
git clone git@gitlab.bj.sensetime.com:facedet/codec/SpringDispatcher.git
cd SpringDispatcher/
git checkout codec_nas
cd ..
git clone git@gitlab.bj.sensetime.com:facedet/codec/springnas-lite.git
cd springnas-lite
git checkout codec_nas
cd ..
git clone git@gitlab.bj.sensetime.com:facedet/codec/Deep_Image_Compression.git
cd Deep_Image_Compression/codes/cc/
make
cd -
cd Deep_Image_Compression/
git checkout nas
cd ..
mv Deep_Image_Compression dic
```
## install
```shell script
cd nas
source r0.3.2
conda create --name nas --clone r0.3.2
source activate nas
cd dic
pip install --upgrade pip --user  # upgrade pip
pip install tensorboard --user  # used for display training process
pip install -r requirements.txt --user  # install pip dependences
cd ../SpringDispatcher/
pip install -r requirements.txt --user
cd ../springnas-lite/
pip install -r requirements.txt --user
cd ..
```

## run
```shell script
cd nas
cd dic/spring_helper/train_config_v2/
source r0.3.2
source activate nas
sh run.sh VI_AIC_TITANXP
```

 ## the way to test single file
 ```shell script
cd nas
export PYTHONPATH=springnas-lite:$PYTHONPATH
export PYTHONPATH=SpringDispatcher:$PYTHONPATH
export PYTHONPATH=dic:$PYTHONPATH
srun -p Test --gres=gpu:1 python -m dic.{test_file}
#srun --mpi=pmi2 -p VI_AIC_TITANXP -n8 --gres=gpu:8 --ntasks-per-node=8 --kill-on-bad-exit=1 --job-name=test python -m dic.{test_file} | tee tee.log
```

