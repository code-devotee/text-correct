import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os

class NERDataset(Dataset):
    def __init__(self, train_feature, config, ent2id):
        self.data = train_feature
        self.config = config
        self.ent2id = ent2id
        self.nums = len(train_feature)
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(config.bert_dir))

    def __len__(self):
        return self.nums;

    def __getitem__(self, index):
        return self.data[index]

    # 构造 NER 标签列表
    def get_bieo_data(self, text, data):
        labels = ['O'] * len(text)

        for _, s, e in data:
            labels[s] = 'B-stock-name'
            labels[e] = 'E-stock-name'
            for i in range(s + 1, e):
                labels[i] = 'I-stock-name'

        return labels

    # 个性化数据处理函数
    def collate_fn(self, batch_data):
        max_len = max([len(x['text']) for x in batch_data]) + 2
        input_ids, token_type_ids, attention_mask, labels, raw_text = [], [], [], [], []
        start_ids, end_ids, bieo_labels = [], [], []

        for sample in batch_data:
            text = sample['text']
            label = sample['stock_name']
            encode_dict = self.tokenizer.encode_plus(
                text = list(text),
                # 指定编码后的最大长度
                max_length = max_len, 
                # 将序列补齐到指定长度
                padding = 'max_length', 
                # 超长时截断
                truncation = True, 
                # 表示输入已经按词切分, 对中文字符列表，通常是把每个字符作为一个输入单元
                is_split_into_words = True, 
                # 返回句子类型ID，常用于区分句子A和句子B，例如0表示句子A，1表示句子B，如果只输入一个句子，则返回的全为0
                return_token_type_ids = True, 
                # 返回注意力掩码
                return_attention_mask = True 
            )
            input_ids.append(encode_dict['input_ids'])
            token_type_ids.append(encode_dict['token_type_ids'])
            attention_mask.append(encode_dict['attention_mask'])
            raw_text.append(text)

        for sample in batch_data:
            text = sample['text']
            label = sample['stock_name']
            encode_dict = self.tokenizer._encode_plus(
                text = list(text),
                # 指定编码后的最大长度
                max_length = max_len,
                # 将序列补齐到指定长度
                padding = 'max_length',
                # 超长时截断
                truncation = True,
                # 表示输入已经按词切分, 对中文字符列表，通常是把每个字符作为一个输入单元
                is_split_into_words = True,
                # 返回句子类型ID，常用于区分句子A和句子B，例如0表示句子A，1表示句子B，如果只输入一个句子，则返回的全为0
                return_token_type_ids = True,
                # 返回注意力掩码
                return_attention_mask = True
            )

            bieo_label = self.get_bieo_data(text, label)
            bieo_labels.append(bieo_label)

            start_id, end_id = [0] * len(text), [0] * len(text)
            for _, s, e in label:
                start_id[s] = self.ent2id['stock_name']
                end_id[e] = self.ent2id['stock_name']

            start_id = [0] + start_id + [0]
            end_id = [0] + end_id + [0]

            # 进行padding操作，将每条数据的长度补齐到max_len
            if len(start_id) < max_len:
                start_id = start_id + [0] * (max_len - len(start_id))
                end_id = end_id + [0] * (max_len - len(end_id))

            start_ids.append(start_id)
            end_ids.append(end_id)

        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).float()
        start_ids = torch.tensor(start_ids).long()
        end_ids = torch.tensor(end_ids).long()

        result = ['input_ids', 'token_type_ids', 'attention_mask',
                  'raw_text', 'start_ids', 'end_ids', 'bieo_labels']
        return dict(zip(result, [input_ids, token_type_ids, attention_mask, raw_text,
                                 start_ids, end_ids, bieo_labels]))
