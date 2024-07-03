

from flask import Blueprint, jsonify
from flask import request
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

@explorer_api.route('/fetch_dots_to_img', methods=['POST'])
def generate_from_3dots():
    c_model = MLModelsRegistry.get_model('autocoder')
    g_model = MLModelsRegistry.get_model('impressionist_150')


    if c_model is None:
        return jsonify({'error': 'MLModel not found'}), 404

    if g_model is None:
        return jsonify({'error': 'MLModel not found'}), 404

    point_3d = np.zeros((1,3), dtype=np.float32)
    try:
        point_3d[0,0] = get_3d_dot('1st_dot')
        point_3d[0,1] = get_3d_dot('2nd_dot')
        point_3d[0,2] = get_3d_dot('3rd_dot')
    except ValueError:
        return jsonify({'error': 'Invalid value'}), 402

    logger.info(f'Generated 3d points: {point_3d}')

    point_3d = torch.Tensor(np.array(point_3d).reshape(1,3))

    point_128d = c_model.model_inst.decode(point_3d)
    output = g_model.model_inst(point_128d)
    img = output[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
    normalized_img = (img + 1) / 2 * 255
    normalized_img = normalized_img.astype(np.uint8)

    logger.info('1 image generated from 3d points')

    ## Encode to base64
    images_base64 = convert_to_jpg(normalized_img)

    return jsonify({'data': images_base64})

def convert_to_jpg(img):
    image = Image.fromarray(img, 'RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
def get_3d_dot(name):

    resp = request.args.get(name, type=float)
    if resp is None:
        raise ValueError('invalid value format')

    return resp