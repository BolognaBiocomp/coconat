import sys
import re
import tensorflow as tf
import h5py
import numpy as np
import torch
import esm
from transformers import T5EncoderModel, T5Tokenizer
from transformers.utils import logging
logging.set_verbosity(50)

import subprocess

from . import coconatconfig as cfg

def chunk_sequence(recid, sequence):
    if len(sequence) <= 1022:
        return ["%s_0" % recid], [sequence]
    else:
        chunks, chunk_ids = [], []
        l = int(np.ceil(len(sequence) / 2 ))
        k, e = 2, 1
        while l > 1022:
            e = e + 1
            k = int(2.0**e)
            l = int(np.ceil(l / 2))
        print(k, e, l)
        for i in range(k):
            print(i, i*l, (i+1)*l)
            chunks.append(sequence[i*l:min((i+1)*l, len(sequence))])
            chunk_ids.append("%s_%d" % (recid, i))
        return chunks, chunk_ids

def join_chunks(chunk_ids, embeddings):
    prev = None
    ret = []
    for i, e in enumerate(embeddings):
        if chunk_ids.split("_")[0] != prev:
            ret.append(e)
        else:
            ret[-1] = np.vstack((ret[-1], e))
    return ret

def embed_prot_t5(sequences):
    #device = torch.device(cfg.DEVICE)
    print("Loading pretrained ProtT5 model...", file=sys.stderr)
    model = T5EncoderModel.from_pretrained(cfg.PROT_T5_MODEL)
    tokenizer = T5Tokenizer.from_pretrained(cfg.PROT_T5_MODEL)
    print("Done.", file=sys.stderr)
    seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']) #.to(device)
    attention_mask = torch.tensor(ids['attention_mask']) #.to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

    lengths = [len(sequence) for sequence in sequences]
    ret = []
    for i in range(len(sequences)):
        emb = embedding_repr.last_hidden_state[i,:lengths[i]]
        ret.append(emb.detach().cpu().numpy())
    return ret

def embed_esm(sequences, seq_ids):
    #device = torch.device(cfg.DEVICE)
    print("Loading pretrained ESM1-b model...", file=sys.stderr)
    model, alphabet = esm.pretrained.load_model_and_alphabet(cfg.ESM_MODEL)
    #model.to(device)
    print("Done", file=sys.stderr)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    data = list(zip(seq_ids, sequences))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    #batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    ret = []
    for i, tokens_len in enumerate(batch_lens):
        ret.append(token_representations[i, 1 : tokens_len - 1].detach().cpu().numpy())
    return ret

def predict_register_probability(samples, lengths, work_env):
    model = tf.keras.models.load_model(cfg.COCONAT_REGISTER_MODEL)
    register_out_file = work_env.createFile("registers.", ".tsv")
    pred = model.predict(samples)
    with open(register_out_file, 'w') as rof:
        for i in range(pred.shape[0]):
            for j in range(lengths[i]):
                print(*[str(x) for x in pred[i,j]], "i", sep=" ", file=rof)
            print("", file=rof)
        rof.close()
    return register_out_file

def crf_refine(register_file, work_env):
    crf_stdout = work_env.createFile("crf.stdout.", ".log")
    crf_stderr = work_env.createFile("crf.stderr.", ".log")
    crf_output = work_env.createFile("crf.output.", ".tsv")
    crf_posterior_output_pfx = work_env.createFile("crf.posterior.", "")

    subprocess.call([cfg.CRF_BIN, "-test",
                   "-m", cfg.COCONAT_CRF_MODEL, "-w", "7",
                   "-d", "posterior-viterbi-sum",
                   "-o", crf_output,
                   "-q", crf_posterior_output_pfx, register_file],
                   stdout=open(crf_stdout, 'w'),
                   stderr=open(crf_stderr, 'w'))
    labels, probs = [], []
    lab_prot = ""
    i = 0
    with open(crf_output) as crfo:
        for line in crfo:
            line = line.split()
            if len(line) > 0:
                lab_prot = lab_prot + line[1]
            else:
                labels.append(lab_prot)
                probs.append(np.loadtxt(crf_posterior_output_pfx+"_%d" % i))
                lab_prot = ""
                i = i + 1
        crfo.close()
    return labels, probs

def predict_oligo_state(samples):
    oligo_map = {0:"A",1:"P",2:"3",3:"4"}
    model = tf.keras.models.load_model(cfg.COCONAT_OLIGO_MODEL)
    pred = model.predict(samples)
    probs = []
    oligo_states = []
    for i in range(pred.shape[0]):
        oligo_states.append(oligo_map[np.argmax(pred[i,0])])
        probs.append(np.max(pred[i,0]))
    return oligo_states, probs
