import torch
import torchvision
import torch.optim
import os
from model import model
import numpy as np
from utils import util
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import glob
import time
import statistics


def load_and_process_image(image_path):
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_pil = (np.asarray(img_pil) / 255.0)
        data_lowlight = torch.from_numpy(img_pil).float()
    except:
        img_pil = Image.open(image_path).convert('L')  # Carica come scala di grigi ('L')
        img_pil = (np.asarray(img_pil) / 255.0)
        data_lowlight = torch.from_numpy(img_pil).float()
        # Aggiungi una dimensione di canale e ripetila per ottenere 3 canali
        data_lowlight = data_lowlight.unsqueeze(2).repeat(1, 1, 3)

    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    return data_lowlight


def lowlight(image_path, forward_times_list, opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = load_and_process_image(image_path)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = (time.time() - start)
    forward_times_list.append(end_time)

    image_path_out = image_path.replace(opt["data"]["in_folder"], opt["data"]["out_folder"])
    result_path = image_path_out
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))

    torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
    # get path
    root_path = Path(util.get_parent(os.path.abspath(__file__)))
    opt = util.yaml_parser(Path(root_path, "config", "zero_dce.yaml"))
    forward_times = []
    with torch.no_grad():
        data_folder_in = Path(root_path, "data", opt["data"]["in_folder"])
        file_list = os.listdir(data_folder_in)
        for file_name in file_list:
            test_list = glob.glob(Path(data_folder_in.__str__(), file_name, "*").__str__())
            for image in tqdm(test_list):
                lowlight(image, forward_times, opt)
            print(f"Average forward speed on {file_name} dataset: {1. / statistics.mean(forward_times)}")
            forward_times.clear()
