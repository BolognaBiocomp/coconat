# coconat
Coiled coil predictor

## Requirements

CoCoNat requires Docker Engine to be installed. Please, follow this instructions
for the installation of Docker Engine on a Debian system:

https://docs.docker.com/engine/install/debian/

## Installation

Create a conda environment with Python 3:

```
conda create -n coconat python
conda activate coconat
```

Install dependencies using pip:

```
pip install docker absl-py
```

Clone this repo and cd into the package dir:

```
git clone https://github.com/BolognaBiocomp/coconat
cd coconat
```

Build the Docker image:

```
docker build -t coconat:1.0 .
```

Download the ESM and ProtT5 pLMs (e.g. on ${HOME}):

```
cd
wget https://coconat.biocomp.unibo.it/static/data/coconat-plms.tar.gz
tar xvzf coconat-plms.tar.gz
```

## Run CoCoNat

To run the program use the run_coconat_docker.py script inside the CoCoNat root
directory, providing a FASTA file an output file, and the path where ESM2 and
ProtT5 pLMs are stored, as follows:

```
cd coconat
python run_coconat_docker.py --fasta_file=example-data/example.fasta \
--output_file=example-data/example.tsv --plm_dir=${HOME}/coconat-plms
```
