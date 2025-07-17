# Estuary Remote Sensing
==============================

Repo for estuaries

## Set me up

Some one time repo initialization work. 


Clone repo
```bash
git clone git@github.com:kyledorman/estuary.git
```

#### Mac Dependancy Installation

Install packages through brew
```bash
brew doctor
brew update
brew upgrade
brew install cairo gdal clang uv
```

## Build Me

Repo uses uv

```bash
uv lock
```

### Update Dependencies

To update the dependencies add/delete/update the pyproject.toml file and run
```bash
uv lock
```

### Jupyter Me

To launch jupyter, run
```bash
./start_jupyter.sh
```

### Lint Me

To format code run
```bash
./lint.sh
```

### Tensorboard Me

To launch jupyter, run
```bash
./tensorboard_start.sh
```