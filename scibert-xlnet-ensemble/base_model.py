import torch
from torch import nn
import transformers
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.transformer_model1 = transformers.AutoModel.from_pretrained(
            config.MODEL1_PATH
        )
        self.transformer_model2 = transformers.XLNetModel.from_pretrained(
            config.MODEL2_PATH
        )
        
        self.dropout = nn.Dropout(0.25)
        self.avgpool = nn.AvgPool1d(2, 2)

        self.output = nn.Linear(768, config.NUM_LABELS)

    def forward(
        self,
        input_ids1, 
        attention_mask1=None, 
        input_ids2=None,
        attention_mask2=None, 
        token_type_ids=None
        ):
        
        model1_hidden_states, model1_features = self.transformer_model1(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids,
        )
        
        model2_output = self.transformer_model2(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids,
        )
        model2_hidden_states = model2_output[0]
        model2_features = model2_hidden_states.mean(dim=1)
        
        combined_features = torch.cat((model1_features, model2_features), dim=1)
        combined_features = combined_features.unsqueeze(1)
        pooled = self.avgpool(combined_features)
        pooled = pooled.squeeze(1)
        
        x = self.dropout(pooled)
        x = self.output(x)
        
        return x