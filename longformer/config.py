import transformers

MAX_LEN = 1024
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 3e-05
TOKENIZER = transformers.LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
MODEL_PATH = "../output/model.bin"