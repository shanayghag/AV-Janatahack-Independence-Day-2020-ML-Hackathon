import torch
import transformers


class RobertaMultiheadClass(torch.nn.Module):
    def __init__(self):
        super(RobertaMultiheadClass, self).__init__()
        self.roberta = transformers.RobertaModel.from_pretrained(
            'roberta-base')
        self.drop = torch.nn.Dropout(0.3)
        self.linear_1 = torch.nn.Linear(1536, 768)
        self.linear_2 = torch.nn.Linear(768, 6)

    def forward(self, ids_1, mask_1, ids_2, mask_2):
        output_1 = self.roberta(ids_1, attention_mask=mask_1)
        output_2 = self.roberta(ids_2, attention_mask=mask_2)

        output_1 = output_1.pooler_output
        output_2 = output_2.pooler_output

        output = torch.cat((output_1, output_2), dim=1)
        output = self.drop(output)
        output = self.linear_1(output)
        output = self.drop(output)
        output = self.linear_2(output)

        return output
