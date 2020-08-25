import torch

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from torch.utils.data import Dataset
from config import config


# preprocessing

def clean_abstract(text):
    text = re.sub('([.,!?()])', r' ', text)
    text = text.split()
    text = [x.strip() for x in text]
    text = [x.replace('\n', ' ').replace('\t', ' ') for x in text]
    sw = set(stopwords.words('english'))
    text = [x for x in text if x not in sw]
    text = ' '.join(text)
    return text
    

def get_texts(df):
    texts = df['ABSTRACT'].apply(clean_abstract)
    texts = texts.values.tolist()
    return texts


def get_labels(df):
    labels = df.iloc[:, 3:].values
    return labels


class TransformerDataset(Dataset):
    def __init__(self, df, indices, set_type=None):
        super(TransformerDataset, self).__init__()

        df = df.iloc[indices]
        self.texts = get_texts(df)
        self.set_type = set_type
        if self.set_type != 'test':
            self.labels = get_labels(df)

        self.tokenizer1 = config.TOKENIZER1
        self.tokenizer2 = config.TOKENIZER2
        self.max_length = config.MAX_LENGTH

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        tokenized1 = self.tokenizer1.encode_plus(
            self.texts[index], 
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids1 = tokenized1['input_ids'].squeeze()
        attention_mask1 = tokenized1['attention_mask'].squeeze()
        
        tokenized2 = self.tokenizer2.encode_plus(
            self.texts[index], 
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids2 = tokenized2['input_ids'].squeeze()
        attention_mask2 = tokenized2['attention_mask'].squeeze()

        if self.set_type != 'test':
            return {
                'model1': {
                    'input_ids': input_ids1.long(),
                    'attention_mask': attention_mask1.long(),
                },
                'model2': {
                    'input_ids': input_ids2.long(),
                    'attention_mask': attention_mask2.long(),
                },
                'labels': torch.Tensor(self.labels[index]).float(),
            }

        return {
            'model1': {
                    'input_ids': input_ids1.long(),
                    'attention_mask': attention_mask1.long(),
                },
            'model2': {
                'input_ids': input_ids2.long(),
                'attention_mask': attention_mask2.long(),
            }
        }