from src.functions_utils import Data_Processor
from src.dataset_utils import NERDataset 
from src.config import Args
from pathlib import Path
import json

args = Args().get_parser()
processor = Data_Processor(args.raw_data_dir)
train, dev = processor.get_lable_data()
path = Path(args.raw_data_dir)
ent2id = json.load(open(path / 'train.json', mode='r', encoding="utf-8"))
print(NERDataset(train[:10], args, ent2id))