import argparse
import yaml
import wespeaker
import glob
import numpy as np
import os
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="path to preprocess.yaml")
args = parser.parse_args()

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

model = wespeaker.load_model(config["path"]["model_folder_path"])
model.set_gpu(0)

data_path = config["path"]["data_path"]
output_path = config["path"]["output_path"]

wav_paths = glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True)

for path in tqdm(wav_paths):
    embedding = model.extract_embedding(path)
    np_embedding = embedding.cpu().numpy()

    out_file = path.replace(data_path, output_path).replace("wav", "npy")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)    
    np.save(out_file, np_embedding)
    
