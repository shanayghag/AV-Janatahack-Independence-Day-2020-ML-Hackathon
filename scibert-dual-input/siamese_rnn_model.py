import torch
from torch import nn
import transformers
from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = transformers.AutoModel.from_pretrained(config.MODEL_PATH)
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AvgPool1d(2, 2)
        self.maxpool = nn.MaxPool1d(2, 2)
        
        self.rnn = nn.GRU(
            input_size=768, 
            hidden_size=128, 
            batch_first=True,
            bidirectional=True,
        )
        
        self.output = nn.Linear(256, config.NUM_LABELS)

    def forward(
        self,
        input_ids_titles, 
        attention_mask_titles=None, 
        input_ids_abstracts=None,
        attention_mask_abstracts=None
        ):
        
        titles_hidden_states, _ = self.model(
            input_ids=input_ids_titles,
            attention_mask=attention_mask_titles
        )
        self.rnn.flatten_parameters()
        titles_rnn_out, _ = self.rnn(titles_hidden_states)
        titles_rnn_feat = titles_rnn_out.mean(dim=1)
        titles_rnn_feat = titles_rnn_feat.unsqueeze(1)
        titles_rnn_feat_pooled = self.avgpool(titles_rnn_feat)
        titles_rnn_feat_pooled = titles_rnn_feat_pooled.squeeze(1)
        
        abstracts_hidden_states, _ = self.model(
            input_ids=input_ids_abstracts,
            attention_mask=attention_mask_abstracts
        )
        self.rnn.flatten_parameters()
        abstracts_rnn_out, _ = self.rnn(abstracts_hidden_states)
        abstracts_rnn_feat = abstracts_rnn_out.mean(dim=1)
        abstracts_rnn_feat = abstracts_rnn_feat.unsqueeze(1)
        abstracts_rnn_feat_pooled = self.avgpool(abstracts_rnn_feat)
        abstracts_rnn_feat_pooled = abstracts_rnn_feat_pooled.squeeze(1)

        combined_features = torch.cat((
            titles_rnn_feat_pooled, 
            abstracts_rnn_feat_pooled), 
            dim=1
        )
        x = self.dropout(combined_features)
        x = self.output(x)
        
        return x