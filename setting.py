import os
import torch

params = {}

### general parameters
params['USE_CUDA'] = torch.cuda.is_available()
params['DEVICE'] = torch.device('cuda:0')

### training parameters
params['NUM_EPOCHS'] = 5
params['LEARNING_RATE'] = 0.0003
params['DROPOUT_PROB'] = 0.35
params['TEST_EXAMPLE_TO_PRINT'] = 5
params['VALID_EXAMPLE_TO_PRINT'] = 5
params['MODEL_NAME'] = "lstm"
params['TRAIN_BATCH_SIZE'] = 4
params['VALID_BATCH_SIZE'] = 4
params['MODEL_SAVE_FREQ'] = 10

### model parameters
params['EMBEDDING_DIM'] = 1024
params['HIDDEN_SIZE'] = 1024
params['NUM_LAYERS'] = 2

