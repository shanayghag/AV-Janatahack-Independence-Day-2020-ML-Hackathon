import torch
from config import config
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score
)


def val(model, device, val_dataloader, criterion):
    
    val_loss = 0
    true, pred = [], []
    
    # set model.eval() every time during evaluation
    model.eval()
    
    for step, batch in enumerate(val_dataloader):
        # unpack the batch contents and push them to the device (cuda or cpu).
        b_input_ids1 = batch['model1']['input_ids'].to(device)
        b_attention_mask1 = batch['model1']['attention_mask'].to(device)
        b_input_ids2 = batch['model2']['input_ids'].to(device)
        b_attention_mask2 = batch['model2']['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # using torch.no_grad() during validation/inference is faster -
        # - since it does not update gradients.
        with torch.no_grad():
            # forward pass
            logits = model(
                b_input_ids1, 
                b_attention_mask1,
                b_input_ids2,
                b_attention_mask2
            )
            
            # calculate loss
            loss = criterion(logits, b_labels)
            val_loss += loss.item()

            # since we're using BCEWithLogitsLoss, to get the predictions -
            # - sigmoid has to be applied on the logits first
            logits = torch.sigmoid(logits)
            logits = np.round(logits.cpu().numpy())
            labels = b_labels.cpu().numpy()
            
            # the tensors are detached from the gpu and put back on -
            # - the cpu, and then converted to numpy in order to -
            # - use sklearn's metrics.

            pred.extend(logits)
            true.extend(labels)

    avg_val_loss = val_loss / len(val_dataloader)
    print('Val loss:', avg_val_loss)
    print('Val accuracy:', accuracy_score(true, pred))

    val_micro_f1_score = f1_score(true, pred, average='micro')
    print('Val micro f1 score:', val_micro_f1_score)
    return val_micro_f1_score


def train(
	model, 
	device, 
	train_dataloader, 
	val_dataloader, 
	criterion, 
	optimizer, 
	scheduler, 
	epoch
	):
    
    # we validate config.N_VALIDATE_DUR_TRAIN times during the training loop
    nv = config.N_VALIDATE_DUR_TRAIN
    temp = len(train_dataloader) // nv
    temp = temp - (temp % 100)
    validate_at_steps = [temp * x for x in range(1, nv + 1)]
    
    train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, 
                                      desc='Epoch ' + str(epoch))):
        # set model.eval() every time during training
        model.train()
        
        # unpack the batch contents and push them to the device (cuda or cpu).
        b_input_ids1 = batch['model1']['input_ids'].to(device)
        b_attention_mask1 = batch['model1']['attention_mask'].to(device)
        b_input_ids2 = batch['model2']['input_ids'].to(device)
        b_attention_mask2 = batch['model2']['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # clear accumulated gradients
        optimizer.zero_grad()

        # forward pass
        logits = model(
            b_input_ids1, 
            b_attention_mask1,
            b_input_ids2,
            b_attention_mask2
        )
        
        # calculate loss
        loss = criterion(logits, b_labels)
        train_loss += loss.item()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        
        # update scheduler
        scheduler.step()

        if step in validate_at_steps:
            print(f'-- Step: {step}')
            _ = val(model, device, val_dataloader, criterion)
    
    avg_train_loss = train_loss / len(train_dataloader)
    print('Training loss:', avg_train_loss)