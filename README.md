# 573 Group Project
Collaborators: Connor Boyle, Martin Horst, Nikitas Tampakis


This code can be run most conveniently through our [Google Colab notebook](https://colab.research.google.com/drive/1Eg4e6SH6St-yCNQBHFqSjk5rKdCs841o?usp=sharing).

### NOTE: Cached Models

The cached models for D4 will remain available for a short period on the UW Google Drive. To download the full 700+ MB trained
models, you can use [this link](https://drive.google.com/drive/folders/1YQD_TNBHXNCf4sZT--a5eN3QG0lLOjql?usp=sharing).

## Training

### Using Development Set

You can train a model using the following command (after activating the [correct environment](#environment)):

```shell
$ python src/train.py --train-file <TRAIN_FILE> --model-directory <MODEL_DIRECTORY>
```

replacing `<TRAIN_FILE>` with the path to the training data and `<MODEL_DIRECTORY>` with the path to where the model
checkpoints will be saved.

The above command will train using the default hyperparameters for our training loop. It will also use a random 10% of
the training data in `<TRAIN_FILE>` file as a per-epoch validation dataset.

### Using Final Test Set

You can train a model using the following command (after activating the [correct environment](#environment)):

```shell
$ python src/train.py --train-file <TRAIN_FILE> --dev-file <DEV_FILE> --model-directory <MODEL_DIRECTORY>
```

replacing `<TRAIN_FILE>` with the path to the training data, `<DEV_FILE>` with the path to the dev dataset file, and
`<MODEL_DIRECTORY>` with the path to where the model checkpoints will be saved.

The above command will train using the default hyperparameters for our training loop. It will also use `<DEV_FILE>` as a
per-epoch validation dataset.

## Data

These represent the maximum tokenized tweet lengths from the BERT tokenizer
for our train, dev, and test files for Spanglish and Hinglish.
E.G.: r, ##t, @, fra, ##lal, ##icio, ##ux, ##xe, t, ##bh, i, have, bad, sides, too, ., when, i, say, bad, it, ', s, ho, ##rri, ##bly, bad, .

- Spanglish_train.conll: 82
- Spanglish_dev.conll: 78
- Spanglish_test_conll_unlabeled.txt: 76
- Hinglish_train_14k_split_conll.txt: 85
- Hinglish_dev_3k_split_conll.txt: 70
- Hinglish_test_unlabeled_conll_updated.txt: 77

## Classifier

The classifier can be run from the shell with the following command:

```shell
$ python src/classify.py --test-file <TEST_FILE> --model-directory <MODEL_DIRECTORY>/<MODEL_INSTANCE> --output-file <OUTPUT_FILE>
```

replacing `<TEST_FILE>` with the path to a testing data file (
e.g. `data/Semeval_2020_task9_data/Spanglish/Spanglish_test_conll_unlabeled.txt`)
and `<OUTPUT_FILE>` with the path to an output file (e.g. `output.txt`)
`<MODEL_DIRECTORY>` with the path to the directory where trained model instances have been saved, and `<MODEL_INSTANCE>`
with any of the (0-indexed) model instances saved before each epoch, or the `FINAL` model instance saved after the last
epoch (NOTE: in our submitted, trained models, only the `FINAL` model instance has been saved).

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
