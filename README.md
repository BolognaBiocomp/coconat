# coconat
Coiled coil predictor

## Requirements

CoCoNat requires Docker Engine to be installed. Please, follow this instructions
for the installation of Docker Engine on a Debian system:

https://docs.docker.com/engine/install/debian/

CoCoNat requires the loading of large pre-trained protein language models (ProtT5 and ESM2)
in memory. We suggest to run CoCoNat on a machine with at least 4 CPU cores and 48GB of RAM.

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

The output file (example-data/example.tsv) looks like the following:

```
ID      RES     CC_CLASS        OligoState      Pi      Pa      Pb      Pc      Pd      Pe      Pf      Pg      PH      POligo
Q99LE1  T       i       i       0.9     0.01    0.04    0.0     0.01    0.03    0.0     0.0     0.0     0.0
Q99LE1  R       i       i       0.83    0.01    0.0     0.07    0.01    0.0     0.07    0.01    0.0     0.0
Q99LE1  L       i       i       0.66    0.01    0.0     0.0     0.15    0.0     0.0     0.17    0.0     0.0
Q99LE1  Q       a       A       0.37    0.5     0.01    0.0     0.01    0.11    0.0     0.0     0.0     0.7615782
Q99LE1  F       b       A       0.24    0.0     0.61    0.01    0.0     0.0     0.14    0.0     0.0     0.7615782
Q99LE1  K       c       A       0.14    0.0     0.0     0.68    0.01    0.0     0.0     0.16    0.0     0.7615782
Q99LE1  I       d       A       0.05    0.15    0.0     0.0     0.8     0.0     0.0     0.0     0.0     0.7615782
Q99LE1  V       e       A       0.02    0.0     0.15    0.0     0.0     0.82    0.0     0.0     0.0     0.7615782
Q99LE1  R       f       A       0.01    0.0     0.0     0.15    0.0     0.0     0.83    0.0     0.0     0.7615782
Q99LE1  V       g       A       0.01    0.0     0.0     0.0     0.15    0.0     0.0     0.84    0.0     0.7615782
Q99LE1  M       a       A       0.0     0.97    0.0     0.0     0.0     0.02    0.0     0.0     0.0     0.7615782
```

You have one row for each residues, and columns are defined as follows:

* ID: protein accession, as reported in the input FASTA file
* RES: residue name
* CC_CLASS: predicted coiled-coil class, a-g for registers, i for non-coiled coil regions
* OligoState: predicted oligomeric state (the same for all residues in the helix). Can be A=antiparallel dimer, P=parallel dimer, 3=trimer, 4=tetramer
* Pi: probability for the residue to be in a non-coiled coil region
* Pa-PH: hepatad repeat registers probabilities
* POligo: probability of the predicted oligomeric state (the same for all residues in the helix)

If coiled coil segment boundaries are already available for your sequences and
you only want to predict their oligomeric state, you can run:

```
python run_coconat_state_docker.py --fasta_file=example-data/example.fasta \
--output_file=example-data/example.tsv --plm_dir=${HOME}/coconat-plms \
--seg_file=example-data/example-seg.tsv
```

where example-seg.tsv file looks like the following:
```
Q99LE1  76  93  abcdefgabcdefgabcd
P16087  60  71  defgabcdefga
P16087  80  91  defgabcdefga
```

The sequence Q99LE1 has a single coiled coil segment from position 76 to 93,
with heptad repeat register abcdefgabcdefgabcd, while the sequence P95883 has
two segments, from position 7 to 18 and from position 26 to 36, both with
heptad annotation defgabcdefga.

## Running outside Docker

If you are not able to use the Docker, it is also possible to run CoCoNat directly using the source code. To do so, you firstly need to create a conda environment and install all dependencies:

```
conda create -n coconat python=3.8
conda activate coconat
```

Install dependencies:

```
python -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir numpy biopython fair-esm transformers[torch] sentencepiece
```

Clone this repo:

```
git clone https://github.com/BolognaBiocomp/coconat
```

Download the ESM and ProtT5 pLMs (e.g. on ${HOME}):

```
cd
wget https://coconat.biocomp.unibo.it/static/data/coconat-plms.tar.gz
tar xvzf coconat-plms.tar.gz
```

After that, move to the CoCaNat package root and open with your preferred editor the coconat/coconatconfig.py file.
Then, you need to modify the following variables:

```
# This need to point the actual CoCoNat package root directory in you machine, e.g. /home/cas/coconat
COCONAT_ROOT = "/app/coconat"
=>
COCONAT_ROOT = "/home/cas/coconat"

# This need to point the actual directory storing ESM2 and ProtT5 models in you machine, e.g (/home/cas/coconat-plms):
COCONAT_PLM_DIR = "/mnt/plms"
=>
COCONAT_PLM_DIR = "/home/cas/coconat-plms"
```

Now, you are able to run the coconat.py script placed in the CoCoNat package root.

For abinitio (coiled-coil helix, registers and oligomeric state) prediction run:
```
python coconat.py abinitio -f example-data/example.fasta -o example-data/example.tsv
```

For oligomeric state prediction run:

```
python coconat.py state -f example-data/example.fasta -s example-data/example-seg.tsv -o example-data/example.tsv
```
