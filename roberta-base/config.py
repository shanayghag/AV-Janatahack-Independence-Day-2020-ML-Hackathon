import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-05
TOKENIZER = transformers.RobertaTokenizer.from_pretrained('roberta-base')
MODEL_PATH = "../output/model.bin"