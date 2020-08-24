import torch
from torch import nn
import transformers
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = transformers.AutoModel.from_pretrained(config.MODEL_PATH)
        self.dropout = nn.Dropout(0.25)
        self.avgpool = nn.AvgPool1d(2, 2)
        self.output = nn.Linear(768, config.NUM_LABELS)

    def forward(
        self,
        input_ids_titles, 
        attention_mask_titles=None, 
        input_ids_abstracts=None,
        attention_mask_abstracts=None
        ):
        
        _, titles_features = self.model(
            input_ids=input_ids_titles,
            attention_mask=attention_mask_titles
        )
        titles_features = titles_features.unsqueeze(1)
        titles_features_pooled = self.avgpool(titles_features)
        titles_features_pooled = titles_features_pooled.squeeze(1)
        
        _, abstracts_features = self.model(
            input_ids=input_ids_abstracts,
            attention_mask=attention_mask_abstracts
        )
        abstracts_features = abstracts_features.unsqueeze(1)
        abstracts_features_pooled = self.avgpool(abstracts_features)
        abstracts_features_pooled = abstracts_features_pooled.squeeze(1)
        
        combined_features = torch.cat((
            titles_features_pooled, 
            abstracts_features_pooled), 
            dim=1
        )
        x = self.dropout(combined_features)
        x = self.output(x)
        
        return x