export PYTHONPATH=..:$PYTHONPATH
WORK_PATH=$(dirname "$0")
TIME=$(date "+%Y%m%d%H%M%S%3N")

python -u test_codec.py --mode "$1" --keep_image "$2" --verbose