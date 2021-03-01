import torch
import transformers


class AlbertClass(torch.nn.Module):
    def __init__(self):
        super(AlbertClass, self).__init__()
        self.albert = transformers.AlbertModel.from_pretrained(
            'albert-base-v2')
        self.drop = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.albert(ids, attention_mask=mask)
        output = outputs.pooler_output
        output = self.drop(output)
        output = self.linear(output)

        return output
