import yaml

def load_yaml(fn):
    """loads a yaml file into a dict"""
    with open(fn,'r') as file_:
        try:
            return yaml.load(file_)
        except RuntimeError as e:
            print "failed to load yaml fille {}, {}\n".format(fn,e)

def save_yaml(fn, data):
    with open(fn,'w') as file_:
        yaml.dump(data,file_, default_flow_style=False)
