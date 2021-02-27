import torch

BATCH_SIZE = 8
TRAIN_DATA = "../input/train"
TEST_DATA = "../input/test"
MODEL_OUTPUT = "../models"
MODEL_IN_FEATURES = 3
MODEL_OUT_FEATURES = 2
TRAIN_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
INFERENCE_DEVICE = "cpu"
EPOCHS = 100
LEARNING_RATE = 0.001
BEST_MODEL = "../models/model_params.pt"
