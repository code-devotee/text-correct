from src.functions_utils import Data_Processor 
from src.config import Args

args = Args().get_parser()
processor = Data_Processor(args.raw_data_dir)
train, dev = processor.get_lable_data()
print(train[:10])
