# List of logs and who should be notified of issues
datas=("b" "c" "s")
degradation="d"

# Look for signs of trouble in each log
for d in ${!datas[@]};
do
  python setup.py run_experiments -d "${datas[$d]}" -t "${degradation}"
done