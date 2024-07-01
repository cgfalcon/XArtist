
import torch
import os
from utils.project_configs_helper import ProjectConfig
import utils.ml_models_def as ml_models_def

def _find_device():
    device = 'cpu'
    # Check if MPS is supported and available
    if torch.backends.mps.is_available():
        print("MPS is available on this device.")
        device = torch.device("mps")  # Use MPS device
    else:
        print("MPS not available, using CPU instead.")
        device = torch.device("cpu")  # Fallback to CPU

    return device


class MLModel:
    '''
    impressionist_150:
      name: Impressionism
      filename: models/generator_model_impressionist_150.pt
      epoch: 150
      device: mps
      framework: pytorch
      style: Impressionism'''
    def __init__(self, model_name, model_config):
        self.model_config_name = model_name
        self.display_name = model_config['name']
        self.model_filename = model_config['filename']
        self.device = model_config['device']
        self.framework = model_config['framework']
        self.style = model_config['style']
        self.model_class = model_config['model_class']
        # load model
        self._load_model()

    def __str__(self):
        return f'{self.model_config_name}: [{self.model_class}, {self.display_name}, {self.style}, {self.framework}]'

    def _load_model(self):

        if hasattr(ml_models_def, self.model_class):
            cls = getattr(ml_models_def, self.model_class)
            instance = cls()
            self.model_inst = instance
        else:
            print(f"Model Class {self.model_class} not found.")
        # model_path = 'models/generator_model_128_151.pt'

        device = _find_device()
        if str(device) != self.device:
            print(f'Incompatible device error. Local device: {device}, model device: {self.device}')
            raise ValueError(f'Incompatible device error. Local device: {self.device}')

        self.device = device

        # Resolve current directory
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_directory, '..', self.model_filename)

        # Load model
        self.model_inst.load_state_dict(torch.load(model_path))
        self.model_inst.to(device)
        self.model_inst.eval()  # S
        print(f'Model {self.model_config_name} loaded')
        print(self.model_inst)


class MLModelsRegistry:

    def __init__(self, project_config: ProjectConfig):
        self.models = project_config['ml_models']

        self.model_insts = {}
        # print(f'Load models from {self.models}')
        for model_conf_name in self.models:
            model_configs = self.models[model_conf_name]
            print(f'Model {model_conf_name}\n\t {model_configs}')
            self.model_insts[model_conf_name] = MLModel(model_conf_name, model_configs)



    def get_model_configs(self):
        return self.models

    def get_model(self, model_conf_name):
        if model_conf_name in self.model_insts:
            return self.model_insts[model_conf_name]
        else:
            return None



MLModelsRegistry = MLModelsRegistry(ProjectConfig)