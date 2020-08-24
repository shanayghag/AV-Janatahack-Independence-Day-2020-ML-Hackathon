import torch
from torch.utils.data import Dataset
from config import config
import re


# preprocessing
def clean_text(text):
    text = text.split()
    text = [x.strip() for x in text]
    text = [x.replace('\n', ' ').replace('\t', ' ') for x in text]
    text = ' '.join(text)
    text = re.sub('([.,!?()])', r' \1 ', text)
    return text
    

def get_texts(df):
    titles = df['TITLE'].apply(clean_text)
    titles = titles.values.tolist()
    abstracts = df['ABSTRACT'].apply(clean_text)
    abstracts = abstracts.values.tolist()
    return titles, abstracts


def get_labels(df):
    labels = df.iloc[:, 3:].values
    return labels

class TransformerDataset(Dataset):
    def __init__(self, df, indices, set_type=None):
        super(TransformerDataset, self).__init__()

        df = df.iloc[indices]
        self.titles, self.abstracts = get_texts(df)
        self.set_type = set_type
        if self.set_type != 'test':
            self.labels = get_labels(df)

        self.tokenizer = config.TOKENIZER
        self.max_length = config.MAX_LENGTH

    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, index):
        tokenized_titles = self.tokenizer.encode_plus(
            self.titles[index], 
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids_titles = tokenized_titles['input_ids'].squeeze()
        attention_mask_titles = tokenized_titles['attention_mask'].squeeze()
        
        tokenized_abstracts = self.tokenizer.encode_plus(
            self.abstracts[index], 
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids_abstracts = tokenized_abstracts['input_ids'].squeeze()
        attention_mask_abstracts = tokenized_abstracts['attention_mask'].squeeze()

        if self.set_type != 'test':
            return {
                'titles': {
                    'input_ids': input_ids_titles.long(),
                    'attention_mask': attention_mask_titles.long(),
                },
                'abstracts': {
                    'input_ids': input_ids_abstracts.long(),
                    'attention_mask': attention_mask_abstracts.long(),
                },
                'labels': torch.Tensor(self.labels[index]).float(),
            }

        return {
            'titles': {
                'input_ids': input_ids_titles.long(),
                'attention_mask': attention_mask_titles.long(),
            },
            'abstracts': {
                'input_ids': input_ids_abstracts.long(),
                'attention_mask': attention_mask_abstracts.long(),
            }
        }