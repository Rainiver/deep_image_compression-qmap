export PYTHONPATH=..:$PYTHONPATH

srun \
  --mpi=pmi2 -p "$1" \
  -n1 --gres=gpu:1 \
  --ntasks-per-node=1 --cpus-per-task=8 \
  --job-name=batch-compress-"$4"-"$(date "+%Y%m%d%H%M%S%3N")" \
  python -u batch_compress.py \
  --src_dir "$2" --dst_dir "$3" --transform_back \
  --mode "$4" --quality "$5" --verbose
