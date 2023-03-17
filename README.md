# ski-data-robustness-experiments-kr-2023
Experiments for "Are Symbolic Knowledge Injection Techniques Robust Against Data Quality
Degradation?" (KR2023).

## 1. Download datasets
Execute the command ```python -m setup.py load_datasets``` to download datasets from UCI website.
By default, the command will store the original dataset into ```datasets``` folder.

Datasets are not tracked by git, so you first need to execute this command before doing anything else.

### [Wisconsin breast cancer dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29) (BCW)
It represents clinical data of patients.
It consists of 9 categorical ordinal features:
1. Clump Thickness
2. Uniformity of Cell Size
3. Uniformity of Cell Shape
4. Marginal Adhesion
5. Single Epithelial Cell Size
6. Bare Nuclei
7. Bland Chromatin
8. Normal Nucleoli
9. Mitoses

All features have integer values in [1, 10] range.
Class indicates if the cancer is benign or malignant.

### [Primate splice junction gene sequences dataset](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)) (PSJGS)
It represents DNA sequences.
Each sequence consists of 60 bases.
Values of one base can be `a`, `c`, `g`, `t` (adenine, cytosine, guanine, thymine).
Class indicates if a sequence activate a biological process: `exon-intron`, `intron-exon`, `none`.
The dataset comes with its own knowledge base.
Both dataset and knowledge have special symbols in addition to the 4 bases.
These symbols indicate that for a particular position in the sequence more than one value of the 4 basis is allowed.
For this reason, the dataset is binarized (one-hot encoding) in order to represent dna sequences with just the 4 basis.

### [Census income dataset](https://archive.ics.uci.edu/ml/datasets/census+income) (CI)

It represents general person's data and the yearly income (less or above 50,000 USD).
Features are continuous, (nominal and ordinal) categorical and binary.

1. age, continuous (integers)
2. workclass, nominal categorical
3. fnlwgt (final weight), continuous
4. education, nominal categorical
5. education-num, ordinal categorical (integers)
6. marital-status, nominal categorical
7. occupation, nominal categorical
8. relationship, nominal categorical
9. race, nominal categorical
10. sex, binary
11. capital-gain, continuous
12. capital-loss, continuous
13. hours-per-week, continuous
14. native-country, nominal categorical

## 2. Run experiments
Execute the command ```python -m setup.py run_experiments -t [d, n] -d [b, s, c] -p [u, kins, kill, kbann]``` to run experiments.
The -t flag indicates the type of experiments to run: `d` for data drop degradation experiments, `n` for noise experiments.
The -d flag indicates the dataset to use: `b` for breast cancer, `s` for splice junction, `c` for census income.
The -p flag indicates the type of predictor to use: `u` for the uneducated, `kins` for the [KINS](http://ceur-ws.org/Vol-3204/paper_25.pdf) SKI method, `kill` for [KILL](http://ceur-ws.org/Vol-3261/paper5.pdf) SKI method, `kbann` for [KBANN](http://www.aaai.org/Library/AAAI/1990/aaai90-129.php) SKI method.

Some executions are faster than others, for instance in the case of the BCW dataset, instead other experiments are much longer like in the case of CI dataset.

Results are stored in the `results` folder.

## 3. KL divergence
To compute the intensity of the data degradation we rely on a formula that requires the computation of the KL divergence between the original and the degraded dataset.
To execute this computation, run the command ```python -m setup.py run_divergence```.
Results are stored in the `results` folder in the corresponding subfolder (e.g. `results/drop/breast-cancer/divergences/1.csv`).

## 4. Robustness
To compute the robustness of the SKI methods, run the command ```python -m setup.py compute_robustness -t [d, n]```.
The -t flag indicates the type of experiments to run: `d` for data drop degradation experiments, `n` for noise experiments.
Results are stored in the `results` folder in the corresponding subfolder (e.g. `results/drop/breast-cancer/robustness.csv`).
