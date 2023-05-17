# List of logs and who should be notified of issues
datas=("b" "c" "s")
degradations=("d" "n" "l" "m")
predictors=("u" "kins" "kill" "kbann")

# Look for signs of trouble in each log
echo "Running all experiments in parallel..."
for d in ${!datas[@]};
do
  for e in ${!degradations[@]};
  do
    for p in ${!predictors[@]};
    do
      python setup.py run_experiments -d "${datas[$d]}" -t "${degradations[$e]}" -p "${predictors[$p]}" &
    done
  done
done
wait
echo "Python runs completed successfully"