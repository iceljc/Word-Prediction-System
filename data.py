import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split
from torchtext import data
from utils import greedy_decode, lookup_words
from setting import params
from allennlp.modules.elmo import batch_to_ids

##############################
class Batch:

    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, src, trg, src_vocab, trg_vocab_size, src_pad_index=0, trg_pad_index=0, trg_unk_index=0):
        src, src_lengths = src
        self.src = src
        word_dict = src_vocab.itos
        src_words = []
        for i in range(src.shape[0]):
        	tmp = []
        	for j in range(src.shape[1]):
        		idx = src[i][j]
        		tmp.append(word_dict[idx])
        	src_words.append(tmp)

        self.src_idx = batch_to_ids(src_words)
        self.src_lengths = src_lengths
        self.src_mask = (src != src_pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            self.trg_idx = trg #trg[:, :-1]
            self.trg = torch.zeros([self.nseqs, trg_vocab_size]).scatter(1, trg.cpu(), 1)
            self.trg[:, trg_pad_index] = 0
            self.trg[:, trg_unk_index] = 0
            # self.trg_mask = (self.trg != trg_pad_index)
            self.ntokens = self.trg.data.sum().item()

        if params['USE_CUDA']:
            self.src = self.src.cuda()
            self.src_idx = self.src_idx.cuda()
            self.src_mask = self.src_mask.cuda()
            if trg is not None:
                self.trg_idx = self.trg_idx.cuda()
                self.trg = self.trg.cuda()
                # self.trg_mask = self.trg_mask.cuda()

##############################


def rebatch(batch, src_pad_idx, trg_pad_idx, trg_unk_idx, trg_vocab_size, src_vocab):
    return Batch(batch.src, batch.trg, src_vocab, trg_vocab_size, src_pad_idx, trg_pad_idx, trg_unk_idx)


##############################
def wiki_data(SRC_Field, TRG_Field, split_factor=0.7, MIN_FREQ=5, data_dir='/mnt/sdd/iceljc/transformer/data/'):
    fields = [('src', SRC_Field), ('trg', TRG_Field)]

    source_path = os.path.expanduser(data_dir+'source_bbc.txt')
    target_path = os.path.expanduser(data_dir+'target_bbc.txt')
    print("SRC Path: ", source_path)
    print("TRG Path: ", target_path)

    examples = []
    count = 1
    with open(source_path, errors='replace') as source_file, open(target_path, errors='replace') as target_file:
        for normal_line, target_line in zip(source_file, target_file):
            normal_line, target_line = normal_line.strip(), target_line.strip()
            # print(count)
            if count%1000 == 0:
                print("Processed ", count, " sentences...")
            count += 1
            if normal_line != '' and target_line != '':
                examples.append(data.Example.fromlist([normal_line, target_line.split()], fields))

    train_len = round(split_factor*len(examples))
    valid_len = round(((1-split_factor)/2)*len(examples))
    test_len = len(examples) - train_len - valid_len
    print("Training set size: ", train_len)
    print("Validation set size: ", valid_len)
    print("Test set size: ", test_len)

    train_examples, valid_examples, test_examples = random_split(examples, [train_len, valid_len, test_len])

    # train_examples, valid_examples, test_examples = examples, examples, examples

    train_dataset = data.Dataset(train_examples, fields)
    valid_dataset = data.Dataset(valid_examples, fields)
    test_dataset = data.Dataset(test_examples, fields)

    SRC_Field.build_vocab(train_dataset.src, train_dataset.trg, min_freq = MIN_FREQ)
    TRG_Field.build_vocab(train_dataset.trg, min_freq = MIN_FREQ)
    print("Finished processing data")

    return train_dataset, valid_dataset, test_dataset, SRC_Field.vocab, TRG_Field.vocab

##############################


def traverse_data(data_iter, model,
					src_eos_index=None, trg_eos_index=None, 
					src_vocab=None, trg_vocab=None, 
					x_file='new_x.txt', y_file='new_y.txt'):
	
	model.eval()

	BOS_TOKEN = "<s>"
	EOS_TOKEN = "</s>"
	UNK_TOKEN = "<unk>"

	if src_vocab is not None and trg_vocab is not None:
		src_bos_index = src_vocab.stoi[BOS_TOKEN]
		src_eos_index = src_vocab.stoi[EOS_TOKEN]
		src_unk_index = src_vocab.stoi[UNK_TOKEN]
	else:
		src_bos_index = 0
		src_eos_index = 1
		src_unk_index = 2

	new_x = ""
	new_y = ""
	pred_y = ""
	for i, batch in enumerate(data_iter):
		result = greedy_decode(model, batch.src_idx, batch.src_mask, batch.src_lengths)

		for i in range(batch.nseqs):
			src = batch.src[i].cpu().numpy()
			trg_idx = batch.trg_idx[i].cpu().numpy()
			out = result[i]

			if src_eos_index is not None:
				eos_pos = np.where(src==src_eos_index)[0]

			if len(eos_pos) > 0:
				src = src[: eos_pos[0]]

			src = src[1:] if src[0] == src_bos_index else src
			src = src[:-1] if src[-1] == src_eos_index else src

			target_y = ""
			# remain_words = ""
			source_text = ""
			src = np.array([x for x in src if x != src_unk_index])
			target_words = set(lookup_words(trg_idx, vocab = trg_vocab))
			pred_words = set(lookup_words(out, vocab = trg_vocab))
			pred_y += " ".join(word for word in pred_words) + "\n"
			target_y = " ".join(word for word in target_words)
			remain_words = " ".join(word for word in target_words if word not in pred_words)
			source_text = " ".join(lookup_words(src, vocab = src_vocab))
			print("=================================")
			print("Source: ")
			print(source_text)
			print("Target: ")
			print(target_words)
			print("Prediction: ")
			print(pred_words)
			print("=================================")
			if remain_words != "":
				new_y += remain_words + "\n"
				new_x += " ".join(lookup_words(src, vocab = src_vocab)) + "\n"
		with open(x_file, 'w') as XFILE:
			XFILE.write(new_x)
		with open(y_file, 'w') as YFILE:
			YFILE.write(new_y)
		with open('prediction.txt', 'w') as OUT:
			OUT.write(pred_y)






















