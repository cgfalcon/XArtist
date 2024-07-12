
train_configs = {
  # Dataset Config
  "DATASET_PATH": "..",

  # Dumping Config
  "MODEL_DUMPING_PATH": ".",
  "IMG_DUMPING_PATH": ".",
  "IMAGE_DUMPING_FREQUENCY": 1,
  "MODEL_DUMPING_FREQUENCY": 30,
  "LOG_FREQUENCY": 1,

  # Model Config
  "ARCH": "SNGAN128",
  "EPOCHS": 6000,
  "BOTTOM_SIZE": 4,
  "INPUT_DIM": 128,
  "INPUT_CHN": 3,
  "LATENT_DIM": 128,
  "BATCH_SIZE": 10,
  "GEN_FILTERS": 64,
  "DIS_FILTERS": 64,
  "CATEGORY_GENRE": "style",
  "LABEL_FILTERS": [9],
  "GEN_LR": 0.0001,
  "DIS_LR": 0.0003,
  "ADAM_BETA1": 0.0,
  "ADAM_BETA2": 0.999,
  "TORCH_WORKERS": 10,
  "LOSS_FN": "bce", # 'bce' or 'hinge',
  "N_DIS": 1,
  "N_GEN": 1,
}