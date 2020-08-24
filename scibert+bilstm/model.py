import torch
import transformers

class ScibertBiLstmClass(torch.nn.Module):
    def __init__(self):
        super(ScibertBiLstmClass, self).__init__()
        self.scibert = transformers.AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.drop = torch.nn.Dropout(0.3)
        self.lstm = torch.nn.LSTM(
        input_size = 768,
        hidden_size = 128,
        batch_first = True,
        bidirectional= True,
        dropout = 0.3)
        self.linear = torch.nn.Linear(512, 6)
    
    def forward(self, ids, mask):
        output, _ = self.scibert(ids, attention_mask = mask)
        output, _ = self.lstm(output)
        avg_pool = torch.mean(output, 1)
        max_pool, _ = torch.max(output, 1)
        output = torch.cat((avg_pool, max_pool), 1)
        output = self.drop(output)
        output = self.linear(output)

        return output