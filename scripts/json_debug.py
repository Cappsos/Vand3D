
import json
import os

json_path = '/home/nicoc/data/BraTS_3D/meta.json'

# load json file as dict
def load_json_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r') as f:
        return json.load(f)

meta = load_json_file(json_path)

meta_keys = list(meta.keys())

print(len(meta['train']['brain']))


print(f"Keys in the JSON file: {meta_keys}")