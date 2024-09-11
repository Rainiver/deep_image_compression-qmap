work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --job-name=codec-test-$2 \
python -u playground.py \
--root $2 \
--verbose \
--test_quality_entropy \
--test_real_bpp \
--test_lambda4_id 0 \
--test_b 0
