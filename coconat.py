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

def coconat_state(args):
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

    samples = []
    for i in range(len(sequences)):
        samples.append(np.hstack((prot_t5_embeddings[i], esm1b_embeddings[i])))
    samples = tf.keras.utils.pad_sequences(samples, padding="post", dtype="float32")
    labels, probs = [], []
    segs = {}
    for line in open(args.segments):
        line = line.split()
        if line[0] not in segs:
            segs[line[0]] = []
        segs[line[0]].append((int(line[1])-1,int(line[2]),line[3]))

    for i in range(len(sequences)):
        seq_labels, seq_probs = ["i"] * lengths[i], [[1.0, 0.0] for _ in range(lengths[i])]
        for seg in segs.get(seq_ids[i], []):
            for (j,k) in enumerate(range(seg[0], seg[1])):
                seq_labels[k] = seg[2][j]
                seq_probs[k][0] = 0.0
                seq_probs[k][1] = 1.0
        labels.append(seq_labels)
        probs.append(seq_probs)

    cc_segments = []
    oligo_preds = {}
    oligo_samples = []
    for i in range(len(sequences)):
        oligo_preds[i] = (["i"]*lengths[i], [0.0]*lengths[i])
        if len(re.findall("[abcdefg]+","".join(labels[i]))) > 0:
            for m in re.finditer("[abcdefg]+","".join(labels[i])):
                cc_segments.append((i, m.start(), m.end()))
                r = np.zeros(7)
                r["abcdefg".index("".join(labels[i])[m.start()])] = 1.0
                v = np.concatenate((np.mean(samples[i,m.start():m.end(),:], axis=0),
                                    r, np.array([m.end()-m.start()])))
                oligo_samples.append(v)
    oligo_samples = np.array(oligo_samples)

    oligo_states, oligo_probs = utils.predict_oligo_state(oligo_samples)

    with open(args.outfile, 'w') as outf:
        print("ID", "START", "END", "OligoST", "OligoProb", sep="\t", file=outf)
        for k, s in enumerate(cc_segments):
            print(seq_ids[s[0]], s[1]+1, s[2], oligo_states[k], oligo_probs[k], sep="\t", file=outf)
        outf.close()
    work_env.destroy()
    return 0

def coconat_abinitio(args):
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
                v, u = [], []
                for k in range(m.start(), m.end()):
                    if "".join(labels[i])[k] == "a":
                        v.append(samples[i,k,:])
                    elif "".join(labels[i])[k] == "d":
                        u.append(samples[i,k,:])

                v = np.concatenate((np.mean(np.array(v), axis=0),
                                    np.mean(np.array(u), axis=0))
                oligo_samples.append(v)
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

def main():
    DESC="CoCoNat: prediction of coiled coil proteins"
    parser = argparse.ArgumentParser(description=DESC)
    subparsers  = parser.add_subparsers(title = "subcommands", description = "valid subcommands", help = "additional help")
    abinitparser = subparsers.add_parser("abinitio", help = "Predict coiled-coils abinitio from sequence", description = "CoCoNat: abinitio prediction.")
    oligostparser = subparsers.add_parser("state", help = "Predict oligomerization state from user-provided segments", description = "CoCoNat: oligo state prediction.")
    abinitparser.add_argument("-f", "--fasta",
                              help = "The input FASTA file",
                              dest = "fasta", required = True)
    abinitparser.add_argument("-o", "--output",
                              help = "Output file name",
                              dest = "outfile", required = True)
    abinitparser.set_defaults(func=coconat_abinitio)

    oligostparser.add_argument("-f", "--fasta",
                               help = "The input FASTA file",
                               dest = "fasta", required = True)
    oligostparser.add_argument("-s", "--segments",
                               help = "The annotated CC segments, in TSV",
                               dest = "segments", required = True)
    oligostparser.add_argument("-o", "--output",
                               help = "Output file name",
                               dest = "outfile", required = True)
    oligostparser.set_defaults(func=coconat_state)

    args = parser.parse_args()
    ret = args.func(args)
    return ret



if __name__ == "__main__":
    try:
        main()
    except:
        raise
        sys.exit(1)
