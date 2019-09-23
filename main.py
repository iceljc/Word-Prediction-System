import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import spacy
from torchtext import data
from setting import params
from data import wiki_data, rebatch, traverse_data
from model import make_model, SimpleLossCompute
from utils import compute_prec_recall, print_examples, save_model, print_data_info
from init_weight import init_weight

def run_epoch(data_iter, model, loss_compute, trg_vocab, print_every=10):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_nseqs = 0
    print_tokens = 0
    total_loss = 0.
    total_tp = 0.
    total_fp = 0.
    total_fn = 0.
    # global LOWEST_LOSS
    # global MAX_STEPS_SINCE_SAVE
    # global MAX_STEPS_SINCE_SAVE
    # global BREAKOUT

    for i, batch in enumerate(data_iter, 1):
        out = model.forward(batch.src_idx, batch.src_mask, batch.src_lengths)
        loss, true_pos, false_pos, false_neg = loss_compute(out, batch.trg, batch.nseqs)
        total_tp += true_pos
        total_fp += false_pos
        total_fn += false_neg
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        total_nseqs += batch.nseqs

        if model.training and i%print_every == 0:
            elapsed = time.time() - start
            print("Epoch step: %d, Loss: %f, Tokens per Sec: %f" %
                (i, loss/batch.nseqs, print_tokens/elapsed))
            start = time.time()
            print_tokens = 0
    if (total_tp + total_fp) == 0:
    	total_prec_score = 0
    else:
    	total_prec_score = total_tp / (total_tp + total_fp)

    if (total_tp + total_fn) == 0:
    	total_recall_score = 0
    else:
    	total_recall_score = total_tp / (total_tp + total_fn)

    if (total_prec_score+total_recall_score) == 0:
    	total_f1_score = 0
    else:
    	total_f1_score = 2*total_prec_score*total_recall_score/(total_prec_score+total_recall_score)
    
    return total_loss/total_nseqs, total_f1_score, total_prec_score, total_recall_score

##############################
def run_test(data_iter, model, trg_vocab):

    total_loss = 0.
    total_tp = 0.
    total_fp = 0.
    total_fn = 0.
    total_tokens = 0.
    total_nseqs = 0.
    model.eval()
    for i, batch in enumerate(data_iter, 1):
        out = model.forward(batch.src_idx, batch.src_mask, batch.src_lengths)
        total_tokens += batch.ntokens
        y_pred = out.contiguous().view(-1, out.size(-1))
        y_true = batch.trg.float().cuda()

        true_pos, false_pos, false_neg = compute_prec_recall(y_pred, y_true)
        total_tp += true_pos
        total_fp += false_pos
        total_fn += false_neg
        # if (precision*recall).sum().item() == 0:
        #     f1_score = torch.zeros(precision.shape[0])
        # else:
        #     f1_score = 2*precision*recall/(precision+recall)

    total_prec_score = total_tp / (total_tp + total_fp)
    total_recall_score = total_tp / (total_tp + total_fn)
    total_f1_score = 2*total_prec_score*total_recall_score/(total_prec_score+total_recall_score)

    return total_f1_score, total_prec_score, total_recall_score



def train(model, num_epochs, lr, save_path, print_every=10):

    model.cuda()
    criterion = nn.MultiLabelSoftMarginLoss(reduction="sum")
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    train_losses = []
    train_prec_score = []
    train_recall_score = []
    train_f1_score = []
    valid_losses = []
    valid_prec_score = []
    valid_recall_score = []
    valid_f1_score = []
    for epoch in range(num_epochs):
        if (epoch%params['MODEL_SAVE_FREQ'] == 0):
            save_model(model, save_path)
        # print("Epoch: ", epoch)
        model.train()
        train_loss, train_prec, train_recall, train_f1 = run_epoch((rebatch(b, SRC_PAD_INDEX, TRG_PAD_INDEX, TRG_UNK_INDEX, len(TRG.vocab), SRC.vocab) for b in train_iter),
                                                        model,
                                                        SimpleLossCompute(criterion, optim, is_train=True), trg_vocab=TRG.vocab, print_every=print_every)
        print("Epoch %d, Training loss: %f" %(epoch, train_loss))
        train_losses.append(train_loss)
        train_prec_score.append(train_prec)
        train_recall_score.append(train_recall)
        train_f1_score.append(train_f1)

        model.eval()
        with torch.no_grad():
        	valid_loss, valid_prec, valid_recall, valid_f1 = run_epoch((rebatch(b, SRC_PAD_INDEX, TRG_PAD_INDEX, TRG_UNK_INDEX, len(TRG.vocab), SRC.vocab) for b in valid_iter),
                                                        model,
                                                        SimpleLossCompute(criterion, None, is_train=False), trg_vocab=TRG.vocab)
        print("Epoch %d, Validation loss: %f" %(epoch, valid_loss))
        valid_losses.append(valid_loss)
        valid_prec_score.append(valid_prec)
        valid_recall_score.append(valid_recall)
        valid_f1_score.append(valid_f1)
    print("Saving the model...")
    save_model(model, save_path)
    return train_losses, train_prec_score, train_recall_score, train_f1_score, valid_losses, valid_prec_score, valid_recall_score, valid_f1_score

spacy_en = spacy.load('en')
def tokenize_src(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


if __name__ == "__main__":
	# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
	DEVICE = torch.device('cuda:0')

	###################
	print("Model Info:")
	print("Hidden nodes: ", params['HIDDEN_SIZE'])
	print("Num of epochs: ", params['NUM_EPOCHS'])
	print("Dropout probability: ", params['DROPOUT_PROB'])
	print("Embedding size: ", params['EMBEDDING_DIM'])
	print("Learning rate: ", params['LEARNING_RATE'])
	###################
	CWD = os.environ['PWD']

	SAVE_DIR = CWD + "/trained_models/" + str(params['HIDDEN_SIZE']) + "units/" + str(params['NUM_EPOCHS']) + 'epochs/'
	DATA_DIR = "/mnt/sdd/iceljc/transformer/data/"

	if not os.path.isdir(SAVE_DIR):
		os.makedirs(SAVE_DIR)

	MODEL_SAVE_PATH = SAVE_DIR+params['MODEL_NAME']+'_'+str(params['HIDDEN_SIZE'])+'hidden_'+str(params['NUM_EPOCHS'])+'epochs.pt'

	seed = 42
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	######################

	UNK_TOKEN = "<unk>"
	PAD_TOKEN = "<pad>"
	BOS_TOKEN = "<s>"
	EOS_TOKEN = "</s>"
	LOWER = True

	######################
	print("Building data field ...")
	# source and target data field
	SRC = data.Field(sequential=True, init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
    	batch_first=True, lower=LOWER, include_lengths=True, tokenize=tokenize_src,
    	unk_token=UNK_TOKEN)
	
	TRG = data.Field(sequential=True, init_token=None, eos_token=None, pad_token=PAD_TOKEN,
    	batch_first=True, lower=LOWER, include_lengths=False, unk_token=UNK_TOKEN)

	print("Splitting data ...")
	train_data, valid_data, test_data, src_vocab, trg_vocab = wiki_data(SRC, TRG, split_factor=0.7, MIN_FREQ=1, data_dir='/mnt/sdd/iceljc/transformer/data/')
	SRC.vocab = src_vocab
	TRG.vocab = trg_vocab
	SRC_PAD_INDEX = SRC.vocab.stoi[PAD_TOKEN]
	TRG_PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]
	TRG_UNK_INDEX = TRG.vocab.stoi[UNK_TOKEN]


	######################
	print_data_info(train_data, valid_data, test_data, SRC, TRG)
	######################

	train_iter = data.Iterator(train_data, batch_size=params['TRAIN_BATCH_SIZE'], 
    	train=True, sort_within_batch=True, sort_key=lambda x:(len(x.src),len(x.trg)),
    	repeat=False, device=DEVICE)
	valid_iter = data.Iterator(valid_data, batch_size=params['VALID_BATCH_SIZE'], 
    	train=False, sort=False, repeat=False, sort_within_batch=True, 
    	sort_key=lambda x:(len(x.src),len(x.trg)), device=DEVICE)
	test_iter = data.Iterator(test_data, batch_size=1, train=False, sort=False, 
    	repeat=False, device=DEVICE)

	########################
	print("Building the model ...")
	model = make_model(SRC.vocab, len(SRC.vocab), len(TRG.vocab), embed_size=params['EMBEDDING_DIM'], 
    	hidden_size=params['HIDDEN_SIZE'], num_layers=params['NUM_LAYERS'], 
    	dropout=params['DROPOUT_PROB'])
	model.apply(init_weight)

	print("Start training ... ")
	train_losses, train_prec_score, train_recall_score, train_f1_score, \
    valid_losses, valid_prec_score, valid_recall_score, valid_f1_score = train(model, 
    	num_epochs=params['NUM_EPOCHS'], lr=params['LEARNING_RATE'], 
    	save_path=MODEL_SAVE_PATH, print_every=100)

	with open("result/train_loss.txt", "w") as f:
		for i in train_losses:
			f.write("%s\n" % i)
	with open("result/train_prec.txt", "w") as f:
		for i in train_prec_score:
			f.write("%s\n" % i)
	with open("result/train_recall.txt", "w") as f:
		for i in train_recall_score:
			f.write("%s\n" % i)
	with open("result/train_f1.txt", "w") as f:
		for i in train_f1_score:
			f.write("%s\n" % i)
	with open("result/valid_loss.txt", "w") as f:
		for i in valid_losses:
			f.write("%s\n" % i)
	with open("result/valid_prec.txt", "w") as f:
		for i in valid_prec_score:
			f.write("%s\n" % i)
	with open("result/valid_recall.txt", "w") as f:
		for i in valid_recall_score:
			f.write("%s\n" % i)
	with open("result/valid_f1.txt", "w") as f:
		for i in valid_f1_score:
			f.write("%s\n" % i)

	print("Finish training ... ")
	print_examples((rebatch(x, SRC_PAD_INDEX, TRG_PAD_INDEX, TRG_UNK_INDEX, len(TRG.vocab), SRC.vocab) for x in test_iter), model, 
    	num=params['TEST_EXAMPLE_TO_PRINT'], src_vocab=SRC.vocab, 
    	trg_vocab=TRG.vocab)

	########################
	print("Start testing ... ")
	test_f1_score, test_prec_score, test_recall_score = run_test((rebatch(x, SRC_PAD_INDEX, TRG_PAD_INDEX, TRG_UNK_INDEX, len(TRG.vocab), SRC.vocab) for x in test_iter), 
		model, trg_vocab=TRG.vocab)

	print("test precision score: ", test_prec_score)
	print("test recall score: ", test_recall_score)
	print("test f1 score: ", test_f1_score)

	

	# X_FILE = 'new_x.txt'
	# Y_FILE = 'new_y.txt'

	# traverse_data((rebatch(x, SRC_PAD_INDEX, TRG_PAD_INDEX, TRG_UNK_INDEX, len(TRG.vocab), SRC.vocab) for x in test_iter), 
	# 				model, 
	# 				src_vocab = SRC.vocab, trg_vocab = TRG.vocab, 
	# 				x_file = X_FILE, y_file = Y_FILE)
    



    


    
    


    

    
    

    
    

    
    
    

    
    
    

    
    
    


    



    



    

    



