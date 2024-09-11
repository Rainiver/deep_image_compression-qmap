work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
{
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --job-name=autoTest-$2 \
python -u playground.py \
--root $2 \
--verbose \
--test_quality_entropy \
--test_real_bpp \
--out_dir $3 \
--kept $4 \
--test_list $5 \
> $3/tee_test.log 2>&1;
echo ****over**** >> $3/tee_test.log;
}&
