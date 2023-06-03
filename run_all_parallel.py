import os
import subprocess
import argparse


def run_all_in_parallel(coms, files):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    procs = [subprocess.Popen(coms[i], shell=True, stdout=open(files[i], "w")) for i in range(len(commands))]
    for p in procs:
        p.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run all in parallel options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--experiment", type=str, default='train', help="train to run all training, "
                                                                              "kl to run all divergences, "
                                                                              "metric to compute all metrics, "
                                                                              "plot_accs for plotting all accuracies, "
                                                                              "plot_kl for plotting all kl divergences")
    options = parser.parse_args()

    if options.experiment.lower() in ['train', 't']:
        datas = ["b", "c", "s"]
        degradations = ["l"]  # ["d", "n", "l", "m"]
        predictors = ["u", "kins", "kill", "kbann"]
        commands = ["python setup.py run_experiments -d {} -t {} -p {}".format(d, e, p) for d in datas
                    for e in degradations
                    for p in predictors]
        logfiles = ["logs/d={}-e={}-p={}.txt".format(d, e, p) for d in datas for e in degradations for p in predictors]
        run_all_in_parallel(commands, logfiles)
    elif options.experiment.lower() in ['kl', 'divergence']:
        datas = ["b", "c", "s"]
        degradations = ["l"]  # ["d", "n", "l", "m"]
        commands = ["python setup.py run_divergence -d {} -t {}".format(d, e) for d in datas for e in degradations]
        logfiles = ["logs/kl of d={}-e={}.txt".format(d, e) for d in datas for e in degradations]
        run_all_in_parallel(commands, logfiles)
    elif options.experiment.lower() in ['metric', 'robustness']:
        degradations = ["d", "n", "l", "m"]
        commands = ["python setup.py compute_robustness -t {}".format(e) for e in degradations]
        logfiles = ["logs/robustness computation of e={}.txt".format(e) for e in degradations]
        run_all_in_parallel(commands, logfiles)
    elif options.experiment.lower() in ['plot_accs', 'plot_acc']:
        degradations = ["d", "n", "l", "m"]
        commands = ["python setup.py generate_comparative_distribution_curves -t {}".format(e) for e in degradations]
        logfiles = ["logs/plot accuracies of e={}.txt".format(e) for e in degradations]
        run_all_in_parallel(commands, logfiles)
    elif options.experiment.lower() in ['plot_kl', 'plot_divergence']:
        degradations = ["d", "n", "l", "m"]
        commands = ["python setup.py generate_divergences_plots -t {}".format(e) for e in degradations]
        logfiles = ["logs/plot divergences of e={}.txt".format(e) for e in degradations]
        run_all_in_parallel(commands, logfiles)
    else:
        raise ValueError('Option "{}" is not a valid option!'.format(options.experiment))


