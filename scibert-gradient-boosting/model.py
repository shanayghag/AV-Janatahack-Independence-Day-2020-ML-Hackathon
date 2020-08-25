from torch import nn
from transformers import AutoModel
from config import config
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.transformer_model = AutoModel.from_pretrained(
            config.MODEL_PATH
        )
    
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(768, config.NUM_LABELS)

    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        token_type_ids=None,
        ):

        _, transformer_embeddings = self.transformer_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        x = self.dropout(transformer_embeddings)
        x = self.output(x)
        
        return x, transformer_embeddings


def xgbmodel(device):
    # set device for the XGBClassifer
    if device.type == 'cpu':
        tree_method = 'auto'
    else:
        tree_method = 'gpu_hist'

    xgboost_model = OneVsRestClassifier(XGBClassifier(
        tree_method=tree_method,
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        gamma=0.15,
        colsample_bynode=0.15,
        colsample_bytree=0.2,
        reg_lambda=0.25,
        
    ))
    return xgboost_model


def lgbmmodel(device):
    # set device for the LGBMClassifer
    if device.type == 'cpu':
        tree_method = 'auto'
    else:
        tree_method = 'gpu_hist'

    lgbm_model = OneVsRestClassifier(LGBMClassifier(
        tree_method=tree_method
    ))
    return lgbm_model
