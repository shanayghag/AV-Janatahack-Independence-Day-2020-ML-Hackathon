import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-05
TOKENIZER = transformers.RobertaTokenizer.from_pretrained('roberta-base')
MODEL_PATH = "../output/model.bin"