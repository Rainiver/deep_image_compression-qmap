work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
spring.submit arun --mpi=pmi2 -p $1 --gpu -n8 --gres=gpu:8 --ntasks-per-node=8 --job-name=codec-train-$2 \
"python -u playground.py \
--root $2 \
--tensorboard-summary-level slim \
2>&1 | tee $2/tee.log"
