# 573 Group Project
Collaborators: Connor Boyle, Martin Horst, Nikitas Tampakis

The [shared document report](https://www.overleaf.com/project/60666a8f489d2af234461f37) is hosted on Overleaf.

## Training

You can train a model (and vectorizer) using the following command (after activating the [correct
environment](#environment)):

```shell
$ python src/train.py --train-file <TRAIN_FILE> --model-file <MODEL_FILE>
```

replacing `<TRAIN_FILE>` with the path to training data and `<MODEL_FILE>`
with a path to save the model (and vectorizer).

## Data

We will fill this out with more as we see fit
Max tweet length of Spanglish_train.conll: 40
Max tweet length of Spanglish_dev.conll: 42
Max tweet length of Spanglish_test_conll_unlabeled.txt: 40
Max tweet length of Hinglish_train_14k_split_conll.txt: 56
Max tweet length of Hinglish_dev_3k_split_conll.txt: 44
Max tweet length of Hinglish_test_unlabeled_conll_updated.txt: 41

## Classifier

The classifier can be run from the shell with the following command:

```shell
$ python src/classify.py --test-file <TEST_FILE> --model-file <MODEL_FILE> --output-file <OUTPUT_FILE>
```

replacing `<TEST_FILE>` with the path to a testing data file (
e.g. `data/Semeval_2020_task9_data/Spanglish/Spanglish_test_conll_unlabeled.txt`)
and `<OUTPUT_FILE>` with the path to an output file (e.g. `output.txt`) and
`<MODEL_FILE>` with the path to a saved model file.

## Environment

We load and save our Python environment using Conda. You can **load** the
environment for the **first time** by running the following command from the
root of the repository:

```bash
$ conda env create -f=src/environment.yml
```

You can then **activate** the environment with the following command:

```bash
$ conda activate 573
```

To **update** your current environment with a new or changed `environment.yml`
file, run the following command:

```bash
$ conda env update -f=src/environment.yml
```

If you have added or updated any new packages using `conda install`, **save**
them to `environment.yml` using the following command:

```bash
$ conda env export > src/environment.yml
```

then remove the line starting with `prefix:`.