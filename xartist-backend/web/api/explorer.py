

from flask import Blueprint, jsonify
from flask import request
import json
import uuid
import datetime
import time
import cv2

import utils.logging as xartist_logging
from utils.ml_models_registry import MLModelsRegistry, ml_default_device
from utils.constants import *
import torch
import base64
import numpy as np
from PIL import Image
import io
from ISR.models import RDN, RRDN

sr_model = RRDN(weights='gans')

explorer_api = Blueprint('explorer', __name__, url_prefix='/api/explorer')

logger = xartist_logging.app_logger

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


seed_seq = 1
# torch.manual_seed(seed_seq)
corners_2d = torch.zeros((4, LATENT_DIM))
p1 = torch.rand(1, LATENT_DIM, device=ml_default_device)
p2 = find_farest_point(p1, ml_default_device)
corners_2d[0] = -1 + 2 * p1
corners_2d[1] = corners_2d[0] * -1
# torch.manual_seed(seed_seq * 100)
corners_2d[2] = -1 + 2 * p2
corners_2d[3] = corners_2d[2] * -1


def reverse_to_hd(ratio1, ratio2, device):
    # notes page 26
    ratio1 = torch.tensor(ratio1).float()
    ratio2 = torch.tensor(ratio2).float()

    ratio = torch.zeros((1, 4)).float()
    ratio[0, 0] = (1 - ratio1) * (1 - ratio2) / 4
    ratio[0, 1] = (1 + ratio1) * (1 - ratio2) / 4
    ratio[0, 2] = (1 + ratio1) * (1 + ratio2) / 4
    ratio[0, 3] = (1 - ratio1) * (1 + ratio2) / 4

    aa = torch.matmul(ratio, corners_2d).to(device)
    return aa


# generate 10000 3axis dots
@explorer_api.route('/fetch_dots', methods=['POST'])
def show_3dots():
    mlmodel = MLModelsRegistry.get_model('autocoder')
    if mlmodel is None:
        return jsonify({'success': False, 'error': 'MLModel not found'}), 404
    logger.info(f'Using model {str(mlmodel)}')

    dots = (torch.rand(3000, 128)-0.5)*2


    dots_3d = mlmodel.model_inst.encode(dots).detach().numpy()

    ## Encode to base64

    logger.info(f'Generated 2d dots: {np.shape(dots_3d)}')

    return jsonify({'success': True, 'data': dots_3d.tolist()})


@explorer_api.route('/fetch_dots_to_img', methods=['POST'])
def generate_from_3dots():
    # c_model = MLModelsRegistry.get_model('autocoder')
    ts_start = time.time()

    default_model = 'gan256_bce_impressionism_600'
    model_key = get_param_value('model_key', type=str, default=default_model)
    g_model = MLModelsRegistry.get_model(model_key)

    # if c_model is None:
    #     return jsonify({'success': False, 'error': 'MLModel not found'}), 404

    if g_model is None:
        return jsonify({'success': False, 'error': 'MLModel not found'}), 404

    # point_3d = np.zeros((1,2), dtype=np.float32)
    ratio1 = -1
    ratio2 = -1
    try:
        ratio1 = get_param_value('1st_dot')
        ratio2 = get_param_value('2nd_dot')
        #point_3d[0,2] = get_3d_dot('3rd_dot')
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid value'}), 402

    point_128d = reverse_to_hd(ratio1, ratio2, g_model.device)

    # logger.info(f'Generated 2d points: {point_128d}')

    # point_128d = c_model.model_inst.decode(point_3d)
    # point_128d = point_128d.to(g_model.device)
    output = g_model.model_inst(point_128d)
    img = output[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
    normalized_img = (img + 1) / 2 * 255
    normalized_img = normalized_img.astype(np.uint8)

    # SR
    # sr_ts = time.time()
    # sr_img = sr_model.predict(normalized_img)
    # logger.info(f'SR time: {time.time() - sr_ts}')

    ## Encode to base64
    images_base64 = convert_to_jpg(normalized_img)

    cost_ts = (time.time() - ts_start) * 1000
    logger.info(f'Image generated in {cost_ts} ms')

    return jsonify({'success': True, 'data': images_base64})


@explorer_api.route('/fetch_dots_to_img_batch', methods=['POST'])
def generate_from_3dots_batch():
    # c_model = MLModelsRegistry.get_model('autocoder')
    ts_start = time.time()

    default_model = 'gan256_bce_impressionism_600'
    model_key = get_param_value('model_key', type=str, default=default_model)
    g_model = MLModelsRegistry.get_model(model_key)

    # if c_model is None:
    #     return jsonify({'success': False, 'error': 'MLModel not found'}), 404

    if g_model is None:
        return jsonify({'success': False, 'error': 'MLModel not found'}), 404

    # point_3d = np.zeros((1,2), dtype=np.float32)
    
    # *** Previous version
    # ratio1 = -1
    # ratio2 = -1

    try:
        
        # *** Added: get x,y coordinates of 2 dots
        x1, y1 = get_param_value('x1'), get_param_value('y1')
        x2, y2 = get_param_value('x2'), get_param_value('y2')
        logger.info(f'Point1 {x1}, {y1}, Point2: {x2}, {y2}')

    # *** Added: more Errors    
    except (KeyError, TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid value'}), 402
    
    p1_128 = reverse_to_hd(x1, y1, g_model.device)
    p2_128 = reverse_to_hd(x2, y2, g_model.device)

    images = gen_images(g_model.model_inst, p1_128, p2_128)

    logger.info(f'Batch Images generated from 2D points, {(time.time() - ts_start) * 1000} ms')

    return jsonify({'success': True, 'data': images})    


def gen_images(g_model, start_point, end_point):
    # token = request.cookies.get('token')

    generated_images = []

    n_sample_points = 30

    trajectory = create_trajectory(start_point, end_point, n_sample_points)

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

    logger.info(f'Generated images: {len(generated_images)}')
    return generated_images


def find_random_point(device):
    farest_point = torch.randn(1, LATENT_DIM).to(device)
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

def get_param_value(name, type=float, default=None):

    resp = request.args.get(name, default=default, type=type)
    if resp is None:
        raise ValueError('invalid value format')

    return resp