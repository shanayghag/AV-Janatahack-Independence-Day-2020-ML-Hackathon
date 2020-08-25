import torch

class CustomDataset:

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.abstract = dataframe.ABSTRACT
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.abstract)

    def __getitem__(self, index):
        abstract = str(self.abstract[index]).lower()
        abstract = " ".join(abstract.split())

        inputs = self.tokenizer.encode_plus(
            abstract,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation = True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
