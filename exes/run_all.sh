# List of logs and who should be notified of issues
datas=("b" "c" "s")
degradations=("d" "n" "l" "m")

# Look for signs of trouble in each log
for d in ${!datas[@]};
do
  for e in ${!degradations[@]};
  do
    python setup.py run_experiments -d "${datas[$d]}" -t "${degradations[$e]}"
  done
done