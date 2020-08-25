import pandas as pd
import numpy as np
import copy

import torch
from torch import optim
from torch.utils.data import DataLoader

import transformers

from config import config
from dataset import *
from engine import *
from model import *


if __name__ == '__main__':

	train_df = pd.read_csv(config.DATA_DIR + 'train.csv')

	# train-val split
	np.random.seed(config.SEED)

	dataset_size = len(train_df)
	indices = list(range(dataset_size))
	split = int(np.floor(config.VALIDATION_SPLIT * dataset_size))
	np.random.shuffle(indices)

	train_indices, val_indices = indices[split:], indices[:split]
	train_indices, val_indices = range(10), range(10)

	# dataset & dataloader
	print('\n--- Creating Dataset')
	train_data = TransformerDataset(train_df, train_indices)
	val_data = TransformerDataset(train_df, val_indices)

	train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE)
	val_dataloader = DataLoader(val_data, batch_size=config.BATCH_SIZE)

	b = next(iter(train_dataloader))
	for k, v in b.items():
	    print(f'{k} shape: {v.shape}')


	# set device
	device = config.DEVICE


	# init model
	print('\n--- Initializing Model')
	model = Model()
	model.to(device);
	print('Initialized.')


	# criterion, optimizer, scheduler
	criterion = nn.BCEWithLogitsLoss()

	# define the parameters to be optmized -
	# - and add regularization
	if config.FULL_FINETUNING:
		param_optimizer = list(model.named_parameters())
		no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
		optimizer_parameters = [
		    {
		        "params": [
		            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
		        ],
		        "weight_decay": 0.001,
		    },
		    {
		        "params": [
		            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
		        ],
		        "weight_decay": 0.0,
		    },
		]
		optimizer = optim.AdamW(optimizer_parameters, lr=config.LR)

	num_training_steps = len(train_dataloader) * config.EPOCHS
	scheduler = transformers.get_linear_schedule_with_warmup(
	    optimizer,
	    num_warmup_steps=0,
	    num_training_steps=num_training_steps
	)


	# training & validation
	print('\n--- Training & Validation')
	max_val_micro_f1_score = float('-inf')
	for epoch in range(config.EPOCHS):
	    train(
	    	model, 
	    	device, 
	    	train_dataloader, 
	    	val_dataloader, 
	    	criterion, 
	    	optimizer, 
	    	scheduler, 
	    	epoch
    	)
	    val_micro_f1_score = val(model, device, val_dataloader, criterion)

	    if config.SAVE_BEST_ONLY:
	        if val_micro_f1_score > max_val_micro_f1_score:
	            best_model = copy.deepcopy(model)
	            best_val_micro_f1_score = val_micro_f1_score

	            torch.save(best_model.state_dict(), config.SAVE_MODEL_PATH + '.pt')

	            print(f'--- Best Model. Val loss: {max_val_micro_f1_score} -> {val_micro_f1_score}')
	            max_val_micro_f1_score = val_micro_f1_score


	# gradient boosting
	# get scibert's best model embeddings
	print('\n--- Get SciBERT best_model Embeddings')
	train_embeddings, train_labels = get_tf_embed(
	    best_model, 
	    device,
	    train_dataloader
	)

	val_embeddings, val_labels = get_tf_embed(
	    best_model, 
	    device,
	    val_dataloader
	)

	# init, train, validate xgboost model
	xgboost_model = xgbmodel(device)
	print('\n--- Fitting XGBClassifer')
	xgb_val_micro_f1_score = train_val_grad_boost(
		xgboost_model,
	    train_embeddings, 
	    train_labels, 
	    val_embeddings, 
	    val_labels
	)
	print('XGBoost Val micro f1 score:', xgb_val_micro_f1_score)

	# init, train, validate lightgbm model
	lgbm_model = lgbmmodel(device)
	print('\n--- Fitting LGBMClassifer')
	lgbm_val_micro_f1_score = train_val_grad_boost(
		lgbm_model,
	    train_embeddings, 
	    train_labels, 
	    val_embeddings, 
	    val_labels
	)
	print('LGBM Val micro f1 score:', lgbm_val_micro_f1_score)


