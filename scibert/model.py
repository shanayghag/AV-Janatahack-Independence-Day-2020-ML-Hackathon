from torch import nn
import transformers
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.transformer_model = transformers.AutoModel.from_pretrained(
            config.MODEL_PATH
        )
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(768, config.NUM_LABELS)

    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        token_type_ids=None
        ):

        _, o2 = self.transformer_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        x = self.dropout(o2)
        x = self.output(x)
        
        return x