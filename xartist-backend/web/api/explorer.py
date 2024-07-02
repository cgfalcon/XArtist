

from flask import Blueprint, jsonify
import json
import uuid
import datetime
import time

import utils.logging as xartist_logging
from utils.ml_models_registry import MLModelsRegistry
from utils.constants import *
import torch
import base64
import numpy as np
from PIL import Image
import io


explorer_api = Blueprint('explorer', __name__, url_prefix='/api/explorer')

logger = xartist_logging.app_logger


# generate 10000 3axis dots
@explorer_api.route('/fetch_dots', methods=['POST'])
def show_3dots():
    mlmodel = MLModelsRegistry.get_model('autocoder')
    if mlmodel is None:
        return jsonify({'error': 'MLModel not found'}), 404
    logger.info(f'Using model {str(mlmodel)}')

    dots = (torch.rand(10000, 128)-0.5)*2


    dots_3d = mlmodel.model_inst.encode(dots).detach().numpy()

    ## Encode to base64

    logger.info(f'Generated 3d dots: {np.shape(dots_3d)}')

    return jsonify({'data': dots_3d.tolist()})

@explorer_api.route('/fetch_dots', methods=['POST'])
def generate_from_3dots():
    mlmodel = MLModelsRegistry.get_model('autocoder')
    if mlmodel is None:
        return jsonify({'error': 'MLModel not found'}), 404
    logger.info(f'Using model {str(mlmodel)}')
    return jsonify({'data': []})