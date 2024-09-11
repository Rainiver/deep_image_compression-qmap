work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
spring.submit arun -s --gpu -n$1 --job-name=codec-train-$2 \
"python -u -W ignore playground.py \
--root $2 \
2>&1 | tee $2/tee.log"
