

# 使用Bert，上面直接接一个biaffine

from torch import nn
from fastNLP.modules.dropout import TimestepDropout
import torch

from fastNLP.models.biaffine_parser import ArcBiaffine, LabelBilinear, BiaffineParser
import torch.nn.functional as F


class BertParser(BiaffineParser):
    def __init__(self, embed, num_label, arc_mlp_size=500, label_mlp_size=100, dropout=0.5, use_greedy_infer=False, app_index=0):
        super(BiaffineParser, self).__init__()

        self.embed = embed

        self.mlp = nn.Sequential(nn.Linear(self.embed.embed_size, arc_mlp_size * 2 + label_mlp_size * 2),
                                          nn.LeakyReLU(0.1),
                                          TimestepDropout(p=dropout),)
        self.arc_mlp_size = arc_mlp_size
        self.label_mlp_size = label_mlp_size
        self.arc_predictor = ArcBiaffine(arc_mlp_size, bias=True)
        self.label_predictor = LabelBilinear(label_mlp_size, label_mlp_size, num_label, bias=True)
        self.use_greedy_infer = use_greedy_infer
        self.reset_parameters()

        self.app_index = app_index
        self.num_label = num_label
        if self.app_index != 0:
            raise ValueError("现在app_index必须等于0")

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for name, m in self.named_modules():
            if 'embed' in name:
                pass
            elif hasattr(m, 'reset_parameters') or hasattr(m, 'init_param'):
                pass
            else:
                for p in m.parameters():
                    if len(p.size())>1:
                        nn.init.xavier_normal_(p, gain=0.1)
                    else:
                        nn.init.uniform_(p, -0.1, 0.1)

    def _forward(self, chars, gold_heads=None, char_labels=None):
        batch_size, max_len = chars.shape

        feats = self.embed(chars)
        mask = chars.ne(0)
        feats = self.dropout(feats)
        feats = self.mlp(feats)
        arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
        arc_dep, arc_head = feats[:,:,:arc_sz], feats[:,:,arc_sz:2*arc_sz]
        label_dep, label_head = feats[:,:,2*arc_sz:2*arc_sz+label_sz], feats[:,:,2*arc_sz+label_sz:]

        arc_pred = self.arc_predictor(arc_head, arc_dep) # [N, L, L]

        if gold_heads is None or not self.training:
            # use greedy decoding in training
            if self.training or self.use_greedy_infer:
                heads = self.greedy_decoder(arc_pred, mask)
            else:
                heads = self.mst_decoder(arc_pred, mask)
            head_pred = heads
        else:
            assert self.training # must be training mode
            if gold_heads is None:
                heads = self.greedy_decoder(arc_pred, mask)
                head_pred = heads
            else:
                head_pred = None
                heads = gold_heads

        batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long, device=chars.device).unsqueeze(1)
        label_head = label_head[batch_range, heads].contiguous()
        label_pred = self.label_predictor(label_head, label_dep) # [N, max_len, num_label]
        # 这里限制一下，只有当head为下一个时，才能预测app这个label
        arange_index = torch.arange(1, max_len+1, dtype=torch.long, device=chars.device).unsqueeze(0)\
            .repeat(batch_size, 1) # batch_size x max_len
        app_masks = heads.ne(arange_index) #  batch_size x max_len, 为1的位置不可以预测app
        app_masks = app_masks.unsqueeze(2).repeat(1, 1, self.num_label)
        app_masks[:, :, 1:] = 0
        label_pred = label_pred.masked_fill(app_masks, float('-inf'))
        if gold_heads is not None:
            res_dict = {'loss':self.loss(arc_pred, label_pred, gold_heads, char_labels, mask)}
        else:
            res_dict = {'label_preds': label_pred.max(2)[1], 'head_preds': head_pred}
        return res_dict

    def forward(self, chars, char_heads, char_labels):
        return self._forward(chars, gold_heads=char_heads, char_labels=char_labels)

    @staticmethod
    def loss(arc_pred, label_pred, arc_true, label_true, mask):
        """
        Compute loss.

        :param arc_pred: [batch_size, seq_len, seq_len]
        :param label_pred: [batch_size, seq_len, n_tags]
        :param arc_true: [batch_size, seq_len]
        :param label_true: [batch_size, seq_len]
        :param mask: [batch_size, seq_len]
        :return: loss value
        """

        batch_size, seq_len, _ = arc_pred.shape
        flip_mask = (mask == 0)
        # _arc_pred = arc_pred.clone()
        _arc_pred = arc_pred.masked_fill(flip_mask.unsqueeze(1), -float('inf'))

        arc_true.data[:, 0].fill_(-1)
        label_true.data[:, 0].fill_(-1)

        arc_nll = F.cross_entropy(_arc_pred.view(-1, seq_len), arc_true.view(-1), ignore_index=-1)
        label_nll = F.cross_entropy(label_pred.view(-1, label_pred.size(-1)), label_true.view(-1), ignore_index=-1)

        return arc_nll + label_nll

    def predict(self, chars):
        """

        max_len是包含root的

        :param chars: batch_size x max_len
        :return:
        """
        res = self._forward(chars, gold_heads=None)
        return res