import argparse
import sys

from Bio import SeqIO

from coconat import coconatconfig as cfg
from coconat import utils
from coconat import workenv

def main():
    DESC="CoCoNat: prediction of coiled coil proteins"
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("-f", "--fasta",
                        help = "The input FASTA file",
                        dest = "fasta", required = True)

    args = parser.parse_args()

    sequences, seq_ids = [], []
    for record in SeqIO.parse(args.fasta, 'fasta'):
        seq_ids.append(record.id)
        sequences.append(str(record.seq))

    prot_t5_embeddings = utils.embed_prot_t5(sequences)
    for m in prot_t5_embeddings:
        print(m.shape)



if __name__ == "__main__":
    try:
        main()
    except:
        raise
        sys.exit(1)
