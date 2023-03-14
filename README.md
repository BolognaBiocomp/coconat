# coconat
Coiled coil predictor


## Installation

Create a conda environment with Python 3:

```
conda create -n coconat python
```

Install dependencies using pip:

```
pip install docker absl-py
```

Clone this repo and cd into the package dir:

```
git clone git@github.com:savojard/coconat.git
cd coconat
```

Build the Docker image:

```
docker build -t coconat:1.0 .
```

Download the ESM and ProtT5 pLMs (e.g. on /home/cas/plms):

```
cd
mkdir plms
cd plms
mkdir esm
cd esm
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
cd ..
git lfs install
git clone https://huggingface.co/Rostlab/prot_t5_xl_uniref50
```

## Run CoCoNat

To run the program use the run_coconat_docker.py script, providing a FASTA file
an output file, and the path were ESM and ProtT5 pLMs are stored, as follows:

```
cd coconat
python run_coconat_docker.py --fasta_file=example-data/example.fasta \
--output_file=example-data/example.tsv --plm_dir=/home/cas/plms
```
