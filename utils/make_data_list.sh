work_path=$(dirname $0)

# spring2
spring.submit arun -n1 --job-name=codec-make_data_list \
"python -u -W ignore make_data_list.py"
