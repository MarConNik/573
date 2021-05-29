# 573 Group Project
Collaborators: Connor Boyle, Martin Horst, Nikitas Tampakis

The [shared document report](https://www.overleaf.com/project/60666a8f489d2af234461f37) is hosted on Overleaf.

### NOTE: Cached Models (D3)

The cached Spanglish model for D3 is committed to this repository. It requires `git-lfs` to download the actual file; by
default, cloning this repository will only download a metadata file of the PyTorch model
file (`outputs/D3/Spanglish-ModelV3.2/pytorch_model.bin`) to your local environment. `git-lfs` is included
in the conda environment. To activate LFS and download the cached model run the following commands:
```
git lfs install
git lfs fetch
git lfs pull
```

If you can't get `git-lfs` to
download the full 700+ MB trained model, you can
use [this link](https://drive.google.com/drive/folders/10U76ymJpjGbqNlExJ5PnelO5nb-45KCd?usp=sharing) to reach a copy of
the same model file on our shared Google Drive.

## Training

You can train a model (and vectorizer) using the following command (after activating the [correct
environment](#environment)):

```shell
$ python src/train.py --train-file <TRAIN_FILE> --model-directory <MODEL_DIRECTORY>
```

replacing `<TRAIN_FILE>` with the path to training data and `<MODEL_DIRECTORY>`
with a path to save the model.

## Data

These represent the maximum tokenized tweet lengths from the BERT tokenizer
for our train, dev, and test files for Spanglish and Hinglish.
E.G.: r, ##t, @, fra, ##lal, ##icio, ##ux, ##xe, t, ##bh, i, have, bad, sides, too, ., when, i, say, bad, it, ', s, ho, ##rri, ##bly, bad, .
Spanglish_train.conll: 82
Spanglish_dev.conll: 78
Spanglish_test_conll_unlabeled.txt: 76
Hinglish_train_14k_split_conll.txt: 85
Hinglish_dev_3k_split_conll.txt: 70
Hinglish_test_unlabeled_conll_updated.txt: 77

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

We load and save our base Python environment using Conda. You can **load** the environment for the **first time** by
running the following command from the root of the repository:

```bash
$ conda env create -f=src/environment.yml
```

You can then **activate** the base environment with the following command:

```bash
$ conda activate 573
```

To **update** your current base environment with a new or changed `environment.yml`
file, run the following command:

```bash
$ conda env update -f=src/environment.yml
```

### Dependencies

On top of the base environment, you will need to install package dependencies from `requirements.txt`
(make sure you have activated the base environment you want to use):

```bash
$ pip install -r src/requirements.txt
```
