# config.py
import os

base_dir = os.path.dirname(os.path.realpath(__file__))

config = {
    'logs': os.path.join(base_dir, 'logs/')
}
datasets_path = {
    'ssc': os.path.join(base_dir, 'datasets/water_quality')
}

epsilon = 1e-8
