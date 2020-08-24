from torch import nn
from transformers import XLNetForSequenceClassification
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.transformer_model = XLNetForSequenceClassification.from_pretrained(
            config.MODEL_PATH,
            num_labels=config.NUM_LABELS
        )

    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        token_type_ids=None
        ):

        logits = self.transformer_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        return logits[0]