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
        abstract = str(self.abstract[index])
        abstract = " ".join(abstract.split())

        inputs_abstract = self.tokenizer.encode_plus(
            abstract,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            pad_to_max_length = True,
            truncation = True
        )

        ids_abstract = inputs_abstract['input_ids']
        mask_abstract = inputs_abstract['attention_mask']

        return{
                'ids_abstract': torch.tensor(ids_abstract, dtype=torch.long),
                'mask_abstract': torch.tensor(mask_abstract, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.float)
            }