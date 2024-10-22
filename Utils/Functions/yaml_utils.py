import ruamel.yaml

def initialize_yaml_parser():
    parser = ruamel.yaml.YAML()
    parser.default_flow_style = False
    parser.boolean_representation = ['False', 'True']
    return parser

def load_yaml_config(yaml_parser, file_path):
    with open(file_path, 'r') as stream:
        config_dict = yaml_parser.load(stream)
    return config_dict

def save_yaml_config(yaml_parser, file_path, config_dict):  
    with open(file_path, 'w') as stream:
        yaml_parser.dump(config_dict, stream)

def convert_list(list):
  s = ruamel.yaml.comments.CommentedSeq(list)
  s.fa.set_flow_style()
  return s

