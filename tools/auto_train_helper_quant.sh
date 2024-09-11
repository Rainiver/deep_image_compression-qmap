work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
{
#srun --mpi=pmi2 -p $1 -n8 --gres=gpu:8 --ntasks-per-node=8 --job-name=$2 \
#python -u -W ignore playground.py \
#--root $2 \
#--quant_cfg ../experiments/integer_configs/warm_w8_a8.yaml \
#2>&1 | tee $2/tee.log;

spring.submit arun -n8 --gpu --job-name=codec-train-$1 \
"python -u -W ignore playground.py --root $1 --quant_cfg ../experiments/integer_configs/warm_w8_a8.yaml --fast-train 2>&1 | tee $1/tee.log";

echo ****over**** >> $1/tee.log;
}&
