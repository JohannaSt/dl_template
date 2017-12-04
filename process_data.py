import os
import argparse
from modules import io

parser = argparse.ArgumentParser()
parser.add_argument('config_file')

args = parser.parse_args()

config_file = os.path.abspath(args.config_file)

config = io.load_yaml(config_file)

####################################
# Get necessary params
####################################
RAW_DATA_DIR = os.path.asbpath(config['RAW_DATA_DIR'])
DATA_DIR     = os.path.abspath(config['DATA_DIR'])

raw_files    = os.listdir(RAW_DATA_DIR)
