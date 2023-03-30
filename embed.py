#!/usr/local/bin/python -W ignore
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import sys
import re
import numpy as np
import tensorflow as tf

from Bio import SeqIO

from coconat import coconatconfig as cfg
from coconat import utils
from coconat import workenv

def embedd(args):
    work_env = workenv.TemporaryEnv(args.outdir)

    sequences, seq_ids, lengths = [], [], []
    chunks, chunk_ids = [], []
    for record in SeqIO.parse(args.fasta, 'fasta'):
        seq_ids.append(record.id)
        sequences.append(str(record.seq))
        lengths.append(len(str(record.seq)))
        r_chunks, r_chunk_ids = utils.chunk_sequence(record.id, str(record.seq))
        chunks.extend(r_chunks)
        chunk_ids.extend(r_chunk_ids)

    #print(chunks)
    #print(chunk_ids)
    prot_t5_embeddings = utils.embed_prot_t5(chunks)
    esm1b_embeddings = utils.embed_esm(chunks, chunk_ids)

    prot_t5_embeddings = utils.join_chunks(chunk_ids, prot_t5_embeddings)
    esm1b_embeddings = utils.join_chunks(chunk_ids, esm1b_embeddings)


    for i in range(len(sequences)):
        e = np.hstack((prot_t5_embeddings[i], esm1b_embeddings[i]))
        outfile = os.path.join(args.outdir, "%s.npy" % seq_ids[i])
        np.save(outfile, e)

    work_env.destroy()
    return 0

def main():
    DESC="CoCoNat: prediction of coiled coil proteins"
    parser = argparse.ArgumentParser(description=DESC)
    subparsers  = parser.add_subparsers(title = "subcommands", description = "valid subcommands", help = "additional help")
    abinitparser = subparsers.add_parser("abinitio", help = "Predict coiled-coils abinitio from sequence", description = "CoCoNat: abinitio prediction.")
    oligostparser = subparsers.add_parser("state", help = "Predict oligomerization state from user-provided segments", description = "CoCoNat: oligo state prediction.")
    parser.add_argument("-f", "--fasta",
                        help = "The input FASTA file",
                        dest = "fasta", required = True)
    parser.add_argument("-o", "--output",
                        help = "Output dir name",
                        dest = "outdir", required = True)

    args = parser.parse_args()
    ret = embedd(args)
    return ret

if __name__ == "__main__":
    try:
        main()
    except:
        raise
        sys.exit(1)
