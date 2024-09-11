work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --job-name=codec-convert-$2 \
python -u playground.py \
--root $2 \
--to_nart \
--nart_out $3

