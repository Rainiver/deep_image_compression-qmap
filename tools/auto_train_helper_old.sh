work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
{
srun --mpi=pmi2 -p $1 -n8 --gres=gpu:8 --ntasks-per-node=8 --job-name=$2 \
python -u -W ignore playground.py \
--root $2 \
--tensorboard-summary-level slim \
--fast-train \
2>&1 | tee $2/tee.log;
echo ****over**** >> $2/tee.log;
}&
