export PYTHONPATH=..:$PYTHONPATH
python export_prob_table.py \
--root $1 \
--entropy-models entropy_pre:scale-only z_entropy_pre:factorized \
--device gpu \
--output $2