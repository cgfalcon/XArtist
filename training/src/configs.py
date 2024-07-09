
train_configs = {
  "ARCH": "SNDCGAN64",
  "EPOCHS": 400,
  "BOTTOM_SIZE": 4,
  "INPUT_DIM": 64,
  "INPUT_CHN": 3,
  "LATENT_DIM": 128,
  "BATCH_SIZE": 64,
  "GEN_FILTERS": 64,
  "DIS_FILTERS": 64,
  "CATEGORY_GENRE": "genre",
  "LABEL_FILTERS": [9],
  "GEN_LR": 0.0005,
  "DIS_LR": 0.0002,
  "TORCH_WORKERS": 3,
  "LOSS_FN": "hinge" # 'bce' or 'hinge'
}