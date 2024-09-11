work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
{
spring.submit arun --mpi=pmi2 -p $1 --gpu -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --job-name=autoTest-$2 \
"python -u -W ignore playground.py \
--root $2 \
--verbose \
--test_quality_entropy \
--test_real_bpp \
--out_dir $3 \
--kept $4 \
--test_list $5 \
> $3/tee_test.log 2>&1";
echo ****over**** >> $3/tee_test.log;
}&
