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

To run the program use the run_coconat_abinitio_docker.py script inside the
CoCoNat root directory, providing a FASTA file an output file, and the path
where ESM2 and ProtT5 pLMs are stored, as follows:

```
cd coconat
python run_coconat_abinitio_docker.py --fasta_file=example-data/example.fasta \
--output_file=example-data/example.tsv --plm_dir=${HOME}/coconat-plms
```

If coiled coil segment boundaries are already available for your sequences and
you only want to predict their oligomeric state, you can run:

```
python run_coconat_state_docker.py --fasta_file=example-data/example.fasta \
--output_file=example-data/example.tsv --plm_dir=${HOME}/coconat-plms \
--seg_file=example-data/example-seg.tsv
```

where example-seg.tsv file looks like the following:
```
Q99LE1  76  93
P95883  7   18
P95883  26  36
```

The sequence Q99LE1 has a single coiled coil segment from position 76 to 93,
while the sequence P95883 has two segments, from position 7 to 18 and from
position 26 to 36.
