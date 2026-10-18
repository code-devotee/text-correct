from torch.nn import nn
import torch
import os
from transformers import BertModel


# 标签平滑的交叉熵
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction="mean", ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)

        if self.reduction == "sum":
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == "mean":
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(
            log_pred, target, reduction=self.reduction, ignore_index=self.ignore_index
        )


class BaseModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, "config.json")
        assert os.path.exists(bert_dir) and os.path.exists(
            config_path
        ), "pretrained bert file does not exists"
        self.bert_module = BertModel.from_pretrained(
            bert_dir, output_hidden_states=True, hidden_dropout_prob=dropout_prob
        )

    @staticmethod
    def _init_weights(blocks, **kwargs):
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(
                        module.weight, mean=0, std=kwargs.pop("initializer_range", 0.02)
                    )
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
