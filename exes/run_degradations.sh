# List of logs and who should be notified of issues
data="b"
degradations=("d" "n" "l" "m")

# Look for signs of trouble in each log
for e in ${!degradations[@]};
  do
    python setup.py run_experiments -d "${data}" -t "${degradations[$e]}"
done