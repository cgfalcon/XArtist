
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


dynamic_block_api = Blueprint('dynamic_block', __name__, url_prefix='/api/dynamic_block')

logger = xartist_logging.app_logger

token_cache = {}

# A flag wither to create start_point
gen_flag = True
start_point = None


# Concurrent Control
max_token_limit = 5

@dynamic_block_api.route('/get_token', methods=['GET'])
def get_token():
    if len(token_cache) >= max_token_limit:
        logger.info('Max token limit reached')
        return jsonify({'error': 'Max token limit reached. Please try again later.'}), 407


    access_token = uuid.uuid4()
    access_token_obj = {
        'access_token': str(access_token),
        'timestamp': time.time(),
    }
    token_cache[access_token] = access_token_obj
    return jsonify({'data': access_token_obj})

@dynamic_block_api.route('/fetch_images', methods=['POST'])
def fetch_images():
    mlmodel = MLModelsRegistry.get_model('impressionist_150')
    if mlmodel is None:
        return jsonify({'error': 'MLModel not found'}), 404
    logger.info(f'Using model {str(mlmodel)}')

    images = gen_images(mlmodel)

    ## Encode to base64
    images_base64 = [convert_to_jpg(img) for img in images]
    logger.info(f'Generated images: {images_base64}')

    return jsonify({'data': images_base64})


def convert_to_jpg(img):
    image = Image.fromarray(img, 'RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def gen_images(ml_model):
    global gen_flag
    global start_point
    device = ml_model.device
    if gen_flag is True:
        start_point = torch.randn(1, LATENT_DIM).to(device)
    # end_point = torch.randn(1, LATENT_DIM).to(device)
    end_point = find_farest_point(start_point, device)
    generated_images = []
    cost_times = []

    n_sample_points = 50

    trajectory = create_trajectory(start_point, end_point, n_sample_points)

    g_model = ml_model.model_inst

    for idx, inter_point in enumerate(trajectory):
        start_ts = time.time()
        output = g_model(inter_point)
        img = output[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        normalized_img = (img + 1) / 2 * 255
        normalized_img = normalized_img.astype(np.uint8)

        generated_images.append(np.ascontiguousarray(normalized_img))
        cost_ts = (time.time() - start_ts) * 1000
        logger.info(f'[{idx}]Image generated in {cost_ts} ms')

    # update start_point
    if gen_flag is True:
        gen_flag = False

    start_point = end_point

    logger.info(f'Generated images: {len(generated_images)}')
    return generated_images


def find_farest_point(point, device):
    max_attempt = 100
    test_point = torch.randn(1, LATENT_DIM).to(device)
    farest_point = test_point
    max_dist = 0
    for i in range(max_attempt):
        test_point = torch.randn(1, LATENT_DIM).to(device)
        dist = torch.linalg.norm(point - test_point)
        if dist > max_dist:
            max_dist = dist
            farest_point = test_point

    logger.info(f'Distance: {torch.linalg.norm(point - farest_point)}')
    return farest_point


def create_trajectory(start, end, n_samples):
    direction = end - start
    trajectory = [start + (i * direction) / (n_samples + 1) for i in range(n_samples + 2)]

    return trajectory

