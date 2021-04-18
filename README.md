# 573 Group Project
Collaborators: Connor Boyle, Martin Horst, Nikitas Tampakis

The [shared document report](https://www.overleaf.com/project/60666a8f489d2af234461f37) is hosted on Overleaf.

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