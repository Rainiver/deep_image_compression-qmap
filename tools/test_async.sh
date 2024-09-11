work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
spring.submit arun -s --gpu -n$1 --job-name=codec-test-$2 \
"python -u -W ignore playground.py \
--root $2 \
--verbose \
--test_quality_entropy \
--test_real_bpp"
