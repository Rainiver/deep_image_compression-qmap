# script for speed evaluation
# turn on --test_time option to enable cuda synchronization
work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
spring.submit run --mpi=pmi2 -p $1 --gpu -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --job-name=codec-test-$2 \
"python -u -W ignore playground.py \
--root $2 \
--verbose \
--test_quality_entropy \
--test_time \
--test_real_bpp"