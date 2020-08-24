import numpy as np
import pandas as pd
from sklearn import metrics

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
import model

if __name__ == "__main__":
    # Data preprocessing
    df = pd.read_csv("../input/data/train.csv")
    df['list'] = df[df.columns[3:]].values.tolist()
    new_df = df[['TITLE', 'ABSTRACT', 'list']].copy()

    # Creating Dataset and Dataloaders
    train_size = 0.8
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = dataset.CustomDataset(train_dataset, config.TOKENIZER, config.MAX_LEN)
    testing_set = dataset.CustomDataset(test_dataset, config.TOKENIZER, config.MAX_LEN)

    train_params = {'batch_size': config.TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': config.VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # GPU check and setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))

    # Object of RobertaMultiheadClass and setting to device
    model = model.RobertaMultiheadClass()
    model.to(device)

    # Model parameters
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

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=3e-5)
    num_training_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Training loop
    best_micro = 0
    for epoch in range(config.EPOCHS):
        engine.train(epoch, model, training_loader, device, optimizer, scheduler)
        outputs, targets = engine.validation(epoch, model, testing_loader, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        if f1_score_micro > best_micro:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_micro = f1_score_micro