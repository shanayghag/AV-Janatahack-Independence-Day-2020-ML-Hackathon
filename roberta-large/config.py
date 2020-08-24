import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 3e-05
TOKENIZER = transformers.RobertaTokenizer.from_pretrained('roberta-large')
MODEL_PATH = "../output/model.bin"