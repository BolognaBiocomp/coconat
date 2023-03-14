#!/usr/local/bin/python -W ignore
import os
import argparse
import sys
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

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
    parser.add_argument("-o", "--output",
                        help = "Output file name",
                        dest = "outfile", required = True)

    args = parser.parse_args()
    work_env = workenv.TemporaryEnv(os.path.dirname(args.outfile).strip())

    sequences, seq_ids, lengths = [], [], []
    for record in SeqIO.parse(args.fasta, 'fasta'):
        seq_ids.append(record.id)
        sequences.append(str(record.seq))
        lengths.append(len(str(record.seq)))

    prot_t5_embeddings = utils.embed_prot_t5(sequences)
    esm1b_embeddings = utils.embed_esm(sequences, seq_ids)
    samples = []
    for i in range(len(sequences)):
        samples.append(np.hstack((prot_t5_embeddings[i], esm1b_embeddings[i])))
    samples = tf.keras.utils.pad_sequences(samples, padding="post", dtype="float32")
    register_file = utils.predict_register_probability(samples, lengths, work_env)
    labels, probs = utils.crf_refine(register_file, work_env)
    with open(args.outfile, 'w') as outf:
        print("ID", "RES", "CC_CLASS", "Pi", "Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "PH", sep="\t", file=outf)
        for i in range(len(sequences)):
            for j in range(lengths[i]):
                print(seq_ids[i], sequences[i][j], labels[i][j], *[round(x,2) for x in list(probs[i][j])], sep="\t", file=outf)
        outf.close()
    work_env.destroy()
    return 0

if __name__ == "__main__":
    try:
        main()
    except:
        raise
        sys.exit(1)
