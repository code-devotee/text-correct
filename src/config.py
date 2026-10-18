import argparse

class Args:

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser.add_argument('--raw_data_dir', default = './data', help = '数据集的存放路径')
        parser.add_argument('--bert_dir', default = './bert-base-chinese', help = '数据集的存放路径')
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
