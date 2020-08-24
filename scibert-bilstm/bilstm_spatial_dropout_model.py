import torch
import transformers

# To use this model, change the model object in train.py file to this class's object
class ScibertBiLstmSpatialDropoutClass(torch.nn.Module):
    def __init__(self):
        super(ScibertBiLstmSpatialDropoutClass, self).__init__()
        self.scibert = transformers.AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.drop = torch.nn.Dropout(0.3)
        
        self.lstm_1 = torch.nn.LSTM(
        input_size = 768,
        hidden_size = 128,
        batch_first = True,
        bidirectional= True,
        num_layers = 1,
        dropout = 0.3)

        self.drop2d = torch.nn.Dropout2d(p = 0.3)
        self.linear = torch.nn.Linear(512, 6)
    
    def forward(self, ids, mask):
        output, _ = self.scibert(ids, attention_mask = mask)
        output = output.permute(0, 2, 1)
        output = self.drop2d(output)
        output = output.permute(0, 2, 1)
        output, _ = self.lstm_1(output)
        avg_pool = torch.mean(output, 1)
        max_pool, _ = torch.max(output, 1)
        output = torch.cat((avg_pool, max_pool), 1)
        output = self.drop(output)
        output = self.linear(output)

        return output