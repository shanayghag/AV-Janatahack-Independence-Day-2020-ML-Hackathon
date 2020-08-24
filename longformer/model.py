import torch
import transformers

class LongformerClass(torch.nn.Module):
    def __init__(self):
        super(LongformerClass, self).__init__()
        self.longformer = transformers.LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.drop = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 6)
    
    def forward(self, ids, mask):
        _, output= self.longformer(ids, attention_mask = mask)
        output = self.drop(output)
        output = self.linear(output)

        return output