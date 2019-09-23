import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time



def compute_prec_recall(y_pred, y_true):

    assert y_pred.shape == y_true.shape

    diff = (torch.sign(y_pred) - y_true).data.cpu().numpy()
    tp = (diff == 0).sum()
    fp = (diff == 1).sum()
    fn = (diff == -2).sum()

    return tp, fp, fn


##############################
def greedy_decode(model, src_idx, src_mask, src_lengths):
    output = []
    with torch.no_grad():
        output = model.forward(src_idx, src_mask, src_lengths)
    output = output.squeeze(0).cpu().numpy()
    output[output<0] = 0
    out = [np.nonzero(row)[0] for row in output]
    # output = output[0]
    return out

##############################

def lookup_words(idx, vocab=None):
    if vocab is not None:
        pad_idx = vocab.stoi["<pad>"]
        unk_idx = vocab.stoi["<unk>"]
        x = [vocab.itos[i] for i in idx if i != pad_idx and i != unk_idx]
    return [str(t) for t in x]

##############################
def print_examples(example_iter, model, num=0, max_len=100,
                    bos_index=1,
                    src_eos_index = None,
                    trg_eos_index = None,
                    src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""
    model.eval()
    count=0

    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"

    if src_vocab is not None and trg_vocab is not None:
        src_bos_index = src_vocab.stoi[BOS_TOKEN]
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_unk_index = trg_vocab.stoi[UNK_TOKEN]
        # trg_bos_index = trg_vocab.stoi[BOS_TOKEN]
        # trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_bos_index = 0
        src_eos_index = 1
        trg_unk_index = 2
        # trg_bos_index = 1
        # trg_eos_index = None

    for i, batch in enumerate(example_iter, 1):
        src = batch.src.cpu().numpy()[0, :]
        trg_idx = batch.trg_idx.cpu().numpy()[0, :]

        # remove </s>
        src = src[1:] if src[0]==src_bos_index else src
        src = src[:-1] if src[-1]==src_eos_index else src
        # trg = trg[:-1] if trg[-1]==trg_eos_index else trg

        result = greedy_decode(model, batch.src_idx, batch.src_mask, batch.src_lengths)
        print()
        print("Example %d" % i)
        print("Source: ", " ".join(lookup_words(src, vocab=src_vocab)))
        print()
        print("Target: ", set(lookup_words(trg_idx, vocab=trg_vocab)))
        print()
        print("Prediction: ", " ".join(lookup_words(result[0], vocab=trg_vocab)))

        count += 1
        if count == num:
            break

##############################

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path))
    torch.save(model, save_path)

##############################




def tokenize_trg(text):
    return text.split()


##############################
def print_data_info(train_data, valid_data, test_data, src_field, trg_field):
    train_targets = {}
    val_targets = {}

    # for elem in train_data:
    #     for word in vars(elem)['trg']:
    #         try: train_targets[word] += 1
    #         except: train_targets[word] = 1
    # for elem in valid_data:
    #     for word in vars(elem)['trg']:
    #         try: val_targets[word] += 1
    #         except: val_targets[word] = 1
    # for key in val_targets:
    #     if key not in train_targets:
    #         print (key)
    #     else:
    #         print (key, val_targets[key], train_targets[key])

    print("Dataset Info")
    print("###############")
    print("Dataset size: ")
    print("train", len(train_data))
    print("valid", len(valid_data))
    print("test", len(test_data), "\n")

    print("First training example:")
    print("src:", " ".join(vars(train_data[0])['src']), "\n")
    print("trg", vars(train_data[0])['trg'], "\n")

    print("Most common words in source vocab: ")
    print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")

    print("Most common words in target vocab: ")
    print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")

    print("First 10 words in source vocab: ")
    print("\n".join('%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")

    print("First 10 words in target vocab: ")
    print("\n".join('%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")

    print("Number of words in source vocab: ", len(src_field.vocab))
    print("Number of words in target vocab: ", len(trg_field.vocab))
    print("###############")





