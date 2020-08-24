import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 3e-05
TOKENIZER = transformers.AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
MODEL_PATH = "../output/model.bin"