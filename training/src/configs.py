
train_configs = {
  # Dataset Config
  "DATASET_PATH": "..",

  # Dumping Config
  "MODEL_DUMPING_PATH": ".",
  "IMG_DUMPING_PATH": ".",
  "IMAGE_DUMPING_FREQUENCY": 2,
  "MODEL_DUMPING_FREQUENCY": lambda epoch: 30 if epoch < 600 else 10, # Dumping model every 30 epochs if current epoch less than 600, otherwise dumping every 10 epochs
  "LOG_FREQUENCY": 4,
  "IMAGE_SAMPLE_WITH_FIXED_POINTS": True, # Wither to use fixed points to generated checkpoint images

  # Training Resume Configs
  "CTN_CONTINUE_TRAINING": False, # Wither to use pre-trained model. True | False
  "CTN_CONTINUE_TRAINING_EPOCH": 300, # The epoch No. to start from
  "CTN_GENERATOR_PATH": "./generator_model_300.pt", # Path to pre-trained generator model
  "CTN_DISCRIMINATOR_PATH": "./discriminator_model_300.pt", # Path to pre-trained discriminator model
  # "CTN_OPTIMIZER_PATH": "",

  # Model Config
  "ARCH": "SNDCGAN256",
  "EPOCHS": 60000,
  "BOTTOM_SIZE": 4,
  "INPUT_DIM": 256,
  "INPUT_CHN": 3,
  "LATENT_DIM": 128,
  "BATCH_SIZE": 256,
  "GEN_FILTERS": 128,
  "DIS_FILTERS": 64,
  "CATEGORY_GENRE": "style",
  "LABEL_FILTERS": [9, 10, 12],
  "GEN_LR": 0.0001,
  "DIS_LR": 0.00005,
  "ADAM_BETA1": 0.5,
  "ADAM_BETA2": 0.999,
  "TORCH_WORKERS": 12,
  "LOSS_FN": "hinge", # 'bce' or 'hinge',
  "N_DIS": 1,
  "N_GEN": 1,
}