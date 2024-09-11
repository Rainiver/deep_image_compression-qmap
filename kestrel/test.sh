work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
#srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --job-name=codec-test-$2 \
python -u ../tools/playground.py \
--root $1 \
--verbose \
--test_quality_entropy \
--test_real_bpp \
--output_act \
--test_zero_pad
