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
    try:
        assert(sum([int(prot_t5_embeddings[i].shape[0] != lengths[i]) for i in range(len(sequences))]) == 0)
    except:
        print([prot_t5_embeddings[i].shape[0] for i in range(len(sequences))])
        print(lengths)
        raise
    try:
        assert(sum([int(esm1b_embeddings[i].shape[0] != lengths[i]) for i in range(len(sequences))]) == 0)
    except:
        print([esm1b_embeddings[i].shape[0] for i in range(len(sequences))])
        print(lengths)
        raise
    samples = []
    for i in range(len(sequences)):
        samples.append(np.hstack((prot_t5_embeddings[i], esm1b_embeddings[i])))
    samples = tf.keras.utils.pad_sequences(samples, padding="post", dtype="float32")
    register_file = utils.predict_register_probability(samples, lengths, work_env)
    labels, probs = utils.crf_refine(register_file, work_env)
    cc_segments = []
    oligo_preds = {}
    oligo_samples = []
    for i in range(len(sequences)):
        oligo_preds[i] = (["i"]*lengths[i], [0.0]*lengths[i])
        if len(re.findall("[abcdefgH]+","".join(labels[i]))) > 0:
            for m in re.finditer("[abcdefgH]+","".join(labels[i])):
                cc_segments.append((i, m.start(), m.end()))
                v = np.mean(samples[i,m.start():m.end(),:], axis=0)
                oligo_samples.append(np.expand_dims(v, axis=0))
    oligo_samples = np.array(oligo_samples)

    oligo_states, oligo_probs = utils.predict_oligo_state(oligo_samples)
    for k, s in enumerate(cc_segments):
        st, pr = oligo_preds[s[0]]
        for j in range(s[1],s[2]):
            st[j] = oligo_states[k]
            pr[j] = oligo_probs[k]
        oligo_preds[s[0]] = (st, pr)

    with open(args.outfile, 'w') as outf:
        print("ID", "RES", "CC_CLASS", "OligoState", "Pi", "Pa", "Pb", "Pc", "Pd", "Pe", "Pf", "Pg", "PH", "POligo", sep="\t", file=outf)
        for i in range(len(sequences)):
            for j in range(lengths[i]):
                print(seq_ids[i], sequences[i][j], labels[i][j], oligo_preds[i][0][j], *[round(x,2) for x in list(probs[i][j])], oligo_preds[i][1][j], sep="\t", file=outf)
        outf.close()
    work_env.destroy()
    return 0

if __name__ == "__main__":
    try:
        main()
    except:
        raise
        sys.exit(1)
