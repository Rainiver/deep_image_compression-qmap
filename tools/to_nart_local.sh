work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
python -u playground.py \
--root $1 \
--to_nart \
--nart_out $2 \
--nart_backend $3

