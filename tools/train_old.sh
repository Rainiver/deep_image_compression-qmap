work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
srun --mpi=pmi2 -p $1 -n8 --gres=gpu:8 --ntasks-per-node=8 --job-name=codec-train-$2 \
python -u -W ignore playground.py \
--root $2 \
2>&1 | tee $2/tee.log 
#srun --mpi=pmi2 -p $1 -n16 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=1 -w SH-IDC1-10-5-34-[131,134] \
