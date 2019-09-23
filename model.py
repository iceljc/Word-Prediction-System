import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from setting import params
from utils import compute_prec_recall
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0, requires_grad=False)
if params['USE_CUDA']:
    elmo = elmo.cuda()

class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many 
	other models.
	"""
	def __init__(self, encoder, src_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.src_embed = src_embed
		self.generator = generator

	def forward(self, src_idx, src_mask, src_lengths): # src_mask is not used here
		encoder_hidden, encoder_final = self.encode(src_idx, src_mask, src_lengths)
		return self.generator.forward(encoder_final)

	def encode(self, src_idx, src_mask, src_lengths):
		return self.encoder(self.src_embed(src_idx), src_mask, src_lengths)


##############################
class Generator(nn.Module):
	"""Define standard linear + softmax generation step."""
	def __init__(self, hidden_size, trg_vocab):
		super(Generator, self).__init__()
		l1_out_size = 3*hidden_size
		self.dropout1 = nn.Dropout(params['DROPOUT_PROB'])
		self.sm_fc1 = nn.Linear(2*hidden_size, l1_out_size, bias=False)
		self.tanh = nn.Tanh()
		self.dropout2 = nn.Dropout(params['DROPOUT_PROB'])
		self.sm_fc2 = nn.Linear(l1_out_size, trg_vocab, bias=False)

	def forward(self, x):
		dropout1 = self.dropout1(x)
		project1 = self.sm_fc1(dropout1)
		tanh = self.tanh(project1)
		dropout2 = self.dropout2(tanh)
		out = self.sm_fc2(dropout2)
		return out

##############################
class Encoder(nn.Module):
	"""Encodes a sequence of word embeddings"""
	def __init__(self, input_size, hidden_size, num_layers=1, dropout_prob=0.5):
		super(Encoder, self).__init__()
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(input_size = input_size,
							hidden_size = hidden_size,
							num_layers = num_layers,
							batch_first = True,
							bidirectional = True)

	def forward(self, x, mask, lengths):
		"""
		Applies a bidirectional LSTM to sequence of embeddings x.
		The input mini-batch x needs to be sorted by length.
		x should have dimensions [batch, seq_len, embed_dim].
		"""
		### x shape: batch_size x seq_len x embedding_dim
		x = x['elmo_representations'][0]

		# print(x.shape)
		packed = pack_padded_sequence(x, lengths, batch_first=True)
		output, final = self.lstm(packed) # final: (h_n, c_n)
		output, _ = pad_packed_sequence(output, batch_first=True)
		final = final[0]
		# we need to manually concatenate the final states for both directions
		if self.num_layers == 1:
			fwd_final = final[0:final.size(0):2]
			bwd_final = final[1:final.size(0):2]

		else:
			num_dirs = 2
			final = final.view(self.num_layers, num_dirs, -1, self.hidden_size)
			fwd_hidden = []
			bwd_hidden = []
			for layer in range(self.num_layers):
				fwd = final[layer, 0: final.size(0):2]
				bwd = final[layer, 1: final.size(0):2]
				fwd_hidden.append(fwd)
				bwd_hidden.append(bwd)
			fwd_final = torch.cat(fwd_hidden, dim=0).mean(dim=0, keepdim=True)
			bwd_final = torch.cat(bwd_hidden, dim=0).mean(dim=0, keepdim=True)

		# hidden_final = torch.cat([fwd_final, bwd_final], dim=2)
		# print("lstm hidden output:", hidden_final.shape)

			# get hidden states from the last layer 
			# fwd_final = final[self.num_layers-1, 0:final.size(0):2]
			# bwd_final = final[self.num_layers-1, 1:final.size(0):2]
		hidden_final = torch.cat([fwd_final, bwd_final], dim=2)
		
		# fwd_final = final[0:final.size(0):2]
		# bwd_final = final[1:final.size(0):2]
		# hidden_final = torch.cat([fwd_final, bwd_final], dim=2) # [num_layers, batch, 2*hidden_size]
		return output, hidden_final


##############################
class SimpleLossCompute:
	"""A simple loss compute and train function."""

	def __init__(self, criterion, opt=None, is_train=False):
		self.criterion = criterion
		self.opt = opt
		self.is_train = is_train

	def __call__(self, y_pred, y_true, norm):

		# print(y_pred.shape, y_true.shape)
		y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
		# y_true = torch.zeros(y_pred.shape).scatter_(1,  y_true_index.cpu(),1).float().cuda()
		loss = self.criterion(y_pred, y_true)
		loss = loss/norm # norm: number of sequences in a batch

		if self.is_train:
			loss.backward()

		if self.opt is not None:
			# self.opt.zero_grad()
			self.opt.step()
			self.opt.zero_grad()

		tmp_loss = loss.item()*norm
		del loss
		torch.cuda.empty_cache()
		true_pos, false_pos, false_neg = compute_prec_recall(y_pred, y_true)

		return tmp_loss, true_pos, false_pos, false_neg


##############################
def make_model(src_vocab, src_vocab_len, tgt_vocab_len, embed_size=256, hidden_size=512, num_layers=2, dropout=0.5):

	# pretrain_embed = nn.Embedding(src_vocab_len, embed_size)
	# pretrain_embed.weight.requires_grad = False
	# # nn.init.xavier_uniform_(pretrain_embed.state_dict()['weight'])
	# word_dict = src_vocab.stoi

	# glove_dir = '/mnt/sdd/iceljc/test/glove_data/'
	# words = []
	# vector = []
	# embedding_index = {}
	# f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))
	# for line in f:
	# 	values = line.split()
	# 	word = values[0]
	# 	coefs = np.asarray(values[1:], dtype = 'float32')
	# 	embedding_index[word] = coefs
	# 	words.append(word)
	# 	vector.append(coefs)
	# f.close()
	# print('Found %s word vectors in GloVe.' % len(embedding_index))

	# count = 0
	# embedding_dim = params['EMBEDDING_DIM']
	# assert embedding_dim == len(vector[0])
	# embedding_matrix = np.zeros((len(src_vocab), embedding_dim))

	# for word in word_dict:
	# 	idx = word_dict[word]
	# 	embedding_vector = embedding_index.get(word)
	# 	if embedding_vector is not None:
	# 		embedding_matrix[idx] = embedding_vector
	# 	else:
	# 		embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
	# 		count += 1
	# print("Number of words in source texts that are not in glove: ", count)
	# print("Embedding matrix size: ", embedding_matrix.shape)
	# print("Filling with embedding matrix ...")
	# pretrain_embed.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})

	pretrain_embed = elmo


	model = EncoderDecoder(
		Encoder(embed_size, hidden_size, num_layers=num_layers, dropout_prob=dropout),
		pretrain_embed,
		Generator(hidden_size, tgt_vocab_len))
	return model













