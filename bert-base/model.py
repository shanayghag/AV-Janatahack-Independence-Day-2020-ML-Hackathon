import torch
import transformers

class BERTBaseClass(torch.nn.Module):
    def __init__(self):
        super(BERTBaseClass, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        # Using the pooled outputs from bert
        _, output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.drop(output_1)
        output = self.linear(output_2)
        return output