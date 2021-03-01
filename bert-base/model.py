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
        outputs = self.bert(ids, attention_mask=mask,
                            token_type_ids=token_type_ids)

        output = outputs.pooler_output
        output = self.drop(output)
        output = self.linear(output)
        return output
