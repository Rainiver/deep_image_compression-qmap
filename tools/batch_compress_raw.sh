export PYTHONPATH=..:$PYTHONPATH

python -u batch_compress.py \
  --src_dir "$1" --dst_dir "$2" --transform_back \
  --mode "$3" --quality "$4" --verbose
