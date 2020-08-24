import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 3e-05
TOKENIZER = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
MODEL_PATH = "../output/model.bin"