export PYTHONPATH=..:$PYTHONPATH
WORK_PATH=$(dirname "$0")
TIME=$(date "+%Y%m%d%H%M%S%3N")

#srun \
#  --mpi=pmi2 -p "$1" \
#  -n1 --gres=gpu:1 \
#  --ntasks-per-node=1 --cpus-per-task=8 \
#  --job-name=codec-test-"$2"-"$TIME" \
#  python -u test_codec.py --mode "$2" --keep_image "$3" --verbose

# spring2
spring.submit arun -n1 --gpu --job-name=codec-ttest-$1 \
"python -u -W ignore test_codec.py --mode $1 --verbose"