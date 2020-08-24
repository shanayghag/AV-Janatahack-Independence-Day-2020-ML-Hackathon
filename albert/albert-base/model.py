import torch
import transformers

class AlbertBaseClass(torch.nn.Module):
    def __init__(self):
        super(AlbertBaseClass, self).__init__()
        self.albert = transformers.AlbertModel.from_pretrained('albert-base-v2')
        self.drop = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 6)
    
    def forward(self, ids, mask):
        _, output= self.albert(ids, attention_mask = mask)
        output = self.drop(output)
        output = self.linear(output)

        return output