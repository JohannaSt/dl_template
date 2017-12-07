import os
import argparse
from modules import io
from modules import vascular_data as sv

parser = argparse.ArgumentParser()
parser.add_argument('config_file')

args = parser.parse_args()

config_file = os.path.abspath(args.config_file)

config = io.load_yaml(config_file)

try:
    sv.mkdir(config['RESULTS_DIR'])
except RuntimeError as e:
    print "Unable to create results directory {}".format(config['RESULTS_DIR'])
    print e

if not config.has_key('TRAIN_PATTERNS') or\
    not config.has_key('VAL_PATTERNS') or\
    not config.has_key('TEST_PATTERNS') or\
    not config.has_key('FILE_LIST'):
    raise RuntimeError("Configuration file {} does not have TRAIN_PATTERNS, or VAL_PATTERNS, or TEST_PATTERNS,\
        or FILE_LIST".format(config_file))

files = open(config['FILE_LIST'],'r').readlines()

trp = config['TRAIN_PATTERNS']
vap = config['VAL_PATTERNS']
tep = config['TEST_PATTERNS']

trf = [f for f in files if any([k in f for k in trp])]
vaf = [f for f in files if any([k in f for k in vap])]
tef = [f for f in files if any([k in f for k in tep])]

f = open(config['TRAIN_FILE_LIST'],'w').writelines(trf)
f = open(config['VAL_FILE_LIST'],'w').writelines(vaf)
f = open(config['TEST_FILE_LIST'],'w').writelines(tef)
