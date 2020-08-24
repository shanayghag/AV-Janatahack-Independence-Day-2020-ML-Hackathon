import torch
import transformers

class RobertaBaseClass(torch.nn.Module):
    def __init__(self):
        super(RobertaBaseClass, self).__init__()
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base')
        self.drop = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 6)
    
    def forward(self, ids, mask):
        _, output= self.roberta(ids, attention_mask = mask)
        output = self.drop(output)
        output = self.linear(output)

        return output