import re
import keras
import h5py
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from . import coconatconfig as cfg

def embed_prot_t5(sequences):
    device = torch.device(cfg.DEVICE)

    model = T5EncoderModel.from_pretrained(cfg.PROT_T5_MODEL)
    tokenizer = T5Tokenizer.from_pretrained(cfg.PROT_T5_MODEL)

    seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

    lengths = [len(sequence) for sequence in sequences]
    ret = []
    for i in range(len(sequences)):
        emb = embedding_repr.last_hidden_state[i,:lengths[i]]
        ret.append(emb.detach().cpu().numpy())
    return ret

def embed_esm(sequences):
    pass

def predict_register_probability(matrix, lengths, work_env):
    model = keras.models.load_model(cfg.COCONAT_REGISTER_MODEL)
    register_out_file = work_env.createFile("registers.", ".tsv")
    pred = model.predict(matrix)
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
