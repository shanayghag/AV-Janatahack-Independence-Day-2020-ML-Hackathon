import torch
import transformers


class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 42
        self.MODEL_PATH = 'allenai/scibert_scivocab_uncased'
        self.SAVE_MODEL_PATH = 'scibertfft_best_model'
        # self.DATA_DIR = '../input/avjanatahackresearcharticlesmlc/av_janatahack_data/'
        self.DATA_DIR = 'drive/My Drive/hackathons/av_janatahack/data/'
        self.NUM_LABELS = 6

        # data
        self.TOKENIZER = transformers.AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.MAX_LENGTH = 320
        self.BATCH_SIZE = 16
        self.VALIDATION_SPLIT = 0.25

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 3e-5
        self.OPTIMIZER = 'AdamW'
        self.CRITERION = 'BCEWithLogitsLoss'
        self.N_VALIDATE_DUR_TRAIN = 3
        self.SAVE_BEST_ONLY = True
        self.EPOCHS = 1

config = Config()