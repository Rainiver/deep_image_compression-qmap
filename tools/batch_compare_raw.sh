export PYTHONPATH=..:$PYTHONPATH

[ ! "$3" ] && map_arg=() || map_arg=(--map "$3")

python -u batch_compare.py \
  --src_dir "$1" --dst_dir "$2" "${map_arg[@]}"
