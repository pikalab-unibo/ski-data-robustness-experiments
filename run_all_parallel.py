import subprocess

datas = ["b", "c", "s"]
degradations = ["d", "n", "l", "m"]
predictors = ["u", "kins", "kill", "kbann"]
commands = ["python setup.py run_experiments -d {} -t {} -p {}".format(d, e, p) for d in datas for e in degradations for
            p in predictors]
print('commands are {}'.format(commands))
logfiles = ["logs/d={}-e={}-p={}.txt".format(d, e, p) for d in datas for e in degradations for p in predictors]
print('logfiles are {}'.format(logfiles))

procs = [subprocess.Popen(commands[i], shell=True, stdout=open(logfiles[i], "w")) for i in range(len(commands))]
for p in procs:
    p.wait()
