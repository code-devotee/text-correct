import json
from pathlib import Path

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


                
