import torch
import transformers


class RobertaLargeClass(torch.nn.Module):
    def __init__(self):
        super(RobertaLargeClass, self).__init__()
        self.roberta = transformers.RobertaModel.from_pretrained(
            'roberta-large')
        self.drop = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(1024, 6)

    def forward(self, ids, mask):
        outputs = self.roberta(ids, attention_mask=mask)
        output = outputs.pooler_output
        output = self.drop(output)
        output = self.linear(output)

        return output
