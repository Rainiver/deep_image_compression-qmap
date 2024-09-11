export PYTHONPATH=..:$PYTHONPATH

[ ! "$4" ] && map_arg=() || map_arg=(--map "$4")

srun \
  --mpi=pmi2 -p "$1" \
  -n1 --gres=gpu:2 \
  --ntasks-per-node=1 --cpus-per-task=8 \
  --job-name=batch-compare-"$(date "+%Y%m%d%H%M%S%3N")" \
  python -u batch_compare.py \
  --src_dir "$2" --dst_dir "$3" "${map_arg[@]}"
