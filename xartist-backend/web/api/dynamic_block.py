
from flask import Blueprint, jsonify, request, g
import json
import uuid
import datetime
import time
import cv2
import torch
import base64
import numpy as np
from PIL import Image
import io
from ISR.models import RDN, RRDN

import utils.logging as xartist_logging
from utils.ml_models_registry import MLModelsRegistry
from utils.constants import *
import web.api.authorization as authorization

dynamic_block_api = Blueprint('dynamic_block', __name__, url_prefix='/api/dynamic_block')

logger = xartist_logging.app_logger

sr_model = RRDN(weights='gans')


@dynamic_block_api.route('/get_models', methods=['GET'])
# @token_required
def get_models():

    ml_lists = []
    # for ml_conf in MLModelsRegistry.models:
    #     print(ml_conf)
    #     ml = {}
        # if
    ml_lists = [{'model_key': ml, 'model_name': MLModelsRegistry.models[ml]['name']} for ml in MLModelsRegistry.models if  MLModelsRegistry.models[ml]['show_for_public']]

    return jsonify({'success': True, 'data': ml_lists})

@dynamic_block_api.route('/fetch_images', methods=['POST'])
# @token_required
def fetch_images():
    data = request.get_json()  # Get JSON data sent with POST request
    model_key = data.get('model', 'impressionist_150')  # Get the model key or default
    mlmodel = MLModelsRegistry.get_model(model_key)
    if mlmodel is None:
        return jsonify({'success': False, 'error': 'MLModel not found'}), 404
    logger.info(f'Using model {str(mlmodel)}')

    images = gen_images(mlmodel)

    ## Encode to base64
    images_base64 = [convert_to_jpg(img) for img in images]

    return jsonify({'success': True, 'data': images_base64})


def measure_classical_methods(ori_image, scale_factor, methods, sigma=1.0, strength=1.3):
    '''
    methods = ['nearest', 'bilinear', 'bicubic', 'lanczos']
    '''
    new_dimensions = (int(ori_image.shape[1] * scale_factor), int(ori_image.shape[0] * scale_factor))
    all_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    start_time = time.time()
    upscaled_image = cv2.resize(ori_image, new_dimensions, interpolation=all_methods[methods])
    blurred_image = cv2.GaussianBlur(upscaled_image, (0, 0), sigma)
    sharpened_image = cv2.addWeighted(upscaled_image, 1 + strength, blurred_image, -strength, 0)
    end_time = time.time()
    process_time = (end_time - start_time) * 1000  # Convert to milliseconds
    logger.info(f'Image enlarge process time: {process_time} ms')
    return sharpened_image

def convert_to_jpg(img):
    enlarged_image = measure_classical_methods(img, 2.0, 'bicubic')

    image = Image.fromarray(enlarged_image, 'RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def gen_images(ml_model):
    # token = request.cookies.get('token')
    token = "111111111"
    asession = authorization.session_store.get_or_gen_session(token)
    if asession is None:
        raise ValueError('Invalid token')

    device = ml_model.device

    session_model_context = asession.get_value(ml_model.model_config_name, default_value = {})
    start_point = session_model_context.get('start_point', None)

    if start_point is None:
        logger.info(f'Session [{asession.token}] created start point {start_point}')
        start_point = torch.randn(1, LATENT_DIM).to(device)
        session_model_context['start_point'] = start_point
    # end_point = torch.randn(1, LATENT_DIM).to(device)
    end_point = find_random_point(device)
    generated_images = []

    n_sample_points = 80

    logger.info(f'Session [{asession.token}], Start points: {start_point}, End point: {end_point}')

    trajectory = create_trajectory(start_point, end_point, n_sample_points)

    g_model = ml_model.model_inst

    for idx, inter_point in enumerate(trajectory):
        start_ts = time.time()
        output = g_model(inter_point)
        img = output[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        normalized_img = (img + 1) / 2 * 255
        normalized_img = normalized_img.astype(np.uint8)

        g_img = np.ascontiguousarray(normalized_img)
        # Resize to 512 * 512
        # SuperResolution
        # sr_start = time.time()
        # sr_img = sr_model.predict(np.array(g_img))
        # logger.info(f'SuperResolution time: {time.time() - start_ts}')
        generated_images.append(g_img)

        cost_ts = (time.time() - start_ts) * 1000
        logger.info(f'[{idx}]Image generated in {cost_ts} ms')

    session_model_context['start_point'] = end_point
    logger.info(f'Session [{asession.token}] created end point')

    asession.put_session_data_kv(ml_model.model_config_name, session_model_context)

    logger.info(f'Generated images: {len(generated_images)}')
    return generated_images

def find_random_point(device):
    random_point = torch.randn(1, LATENT_DIM).to(device)
    return random_point

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


def create_wave_trajectory(start, end, n_samples, amplitude=1, frequency=0.3):
    # Ensure start and end are torch tensors and moved to the appropriate device
    # start = torch.tensor(start, device=device)
    # end = torch.tensor(end, device=device)

    # Linear interpolation for base trajectory
    direction = end - start
    base_trajectory = [start + (i * direction) / (n_samples + 1) for i in range(n_samples + 2)]

    # Create wave pattern
    wave_trajectory = []
    for i, point in enumerate(base_trajectory):
        # Calculate the offset using a sine wave
        offset = amplitude * np.sin(frequency * i)
        wave_point = point + offset
        wave_trajectory.append(wave_point)

    return torch.stack(wave_trajectory)

