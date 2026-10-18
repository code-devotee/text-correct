import json
import logging
from pathlib import Path
import random
import torch
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# 加载模型 & 放置到GPU中(单卡/多卡)
def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    device = torch.device('cpu' if gpu_ids == '-1' else 'cuda:' + gpu_ids[0])

    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict = strict)
    model.to(device)

    # 如果服务器上有多块GPU，采用分布式训练的模式
    if gpu_ids != -1 and len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids = gpu_ids)
    else:
        logger.info(f'Use single gpu in:{gpu_ids}')
    return model, device

def get_model_path_list(base_dir):
    tmp = os.listdir(base_dir)
    tmp = ['checkpoint_{}_epoch.pt'.format(i) for i in range(21)
           if 'checkpoint_{}_epoch.pt'.format(i) in tmp]
    model_lists = [os.path.join(base_dir, x) for x in tmp]
    return model_lists
    

class Data_Processor:
    def __init__(self, path):
        path = Path(path)
        self.path = path / 'train.json' if path.is_dir() else path

    # 将 train.json 中的 json 转换成三元组
    def get_lable_data(self):
        data = json.load(open(self.path, mode='r', encoding="utf-8"))

        res = []
        for item in data:
            text, stock_names = item['text'], item['stock_name']
            labels = []
            for stock_name in stock_names:
                start_id = text.find(stock_name)
                end_id = start_id + len(stock_name) - 1
                labels.append((stock_name, start_id, end_id))
                text.replace(stock_name, '$' * len(stock_name), 1)
            res.append({'text': text, 'stock_name': labels})

        # 90% 作为训练集， 10% 作为验证集
        train = [x for i, x in enumerate(res) if i % 10 != 0]
        dev = [x for i, x in enumerate(res) if i % 10 == 0]
        return train, dev


                
