work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
srun --mpi=pmi2 -p $1 -n8 --gres=gpu:8 --ntasks-per-node=8 --job-name=codec-train-$2 \
python -m cProfile -s tottime playground.py --root $2 > time_consume.txt
