from modules.pipe import CTBxJointPipe
from fastNLP.embeddings import BertEmbedding
from torch import nn
from functools import partial
from models.BertParser import BertParser
from models.metrics import SegAppCharParseF1Metric, CWSMetric
from fastNLP import BucketSampler, Trainer
from torch import optim
from fastNLP import GradientClipCallback, WarmupCallback
from fastNLP import cache_results
import argparse


uniform_init = partial(nn.init.normal_, std=0.02)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['ctb5', 'ctb7', 'ctb9'], default='ctb5')
args = parser.parse_args()

data_name = args.dataset
###################################################hyper
# 需要变动的超参放到这里
lr = 2e-5   # 0.01~0.001
dropout = 0.5  # 0.3~0.6
arc_mlp_size = 500   # 200, 300
encoder = 'bert'
batch_size = 6
update_every = 1
n_epochs = 5

label_mlp_size = 100
####################################################hyper
data_folder = f'../data/{data_name}'
device = 0

@cache_results('caches/{}_bert.pkl'.format(data_name), _refresh=False)
def get_data():
    data = CTBxJointPipe().process_from_file(data_folder)
    data.delete_field('bigrams')
    data.delete_field('trigrams')
    data.delete_field('chars')
    data.rename_field('pre_chars', 'chars')
    data.delete_field('pre_bigrams')
    data.delete_field('pre_trigrams')
    bert_embed = BertEmbedding(data.get_vocab('chars'), model_dir_or_name='cn', requires_grad=True)
    return data, bert_embed

data, bert_embed = get_data()

print(data)
model = BertParser(embed=bert_embed, num_label=len(data.get_vocab('char_labels')), arc_mlp_size=arc_mlp_size,
                   label_mlp_size=label_mlp_size, dropout=dropout,
                   use_greedy_infer=False,
                   app_index=0)

metric1 = SegAppCharParseF1Metric(data.get_vocab('char_labels')['APP'])
metric2 = CWSMetric(data.get_vocab('char_labels')['APP'])
metrics = [metric1, metric2]

optimizer = optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=lr,
                       weight_decay=1e-2)

sampler = BucketSampler(seq_len_field_name='seq_lens')
callbacks = []

warmup_callback = WarmupCallback(schedule='linear')

callbacks.append(warmup_callback)
callbacks.append(GradientClipCallback(clip_type='value', clip_value=5))

trainer = Trainer(data.datasets['train'], model, loss=None, metrics=metrics, n_epochs=n_epochs, batch_size=batch_size,
                  print_every=3,
                 validate_every=-1, dev_data=data.datasets['dev'], save_path=None, optimizer=optimizer,
                 check_code_level=0, metric_key='u_f1', sampler=sampler, num_workers=2, use_tqdm=True,
                 device=device, callbacks=callbacks, update_every=update_every, dev_batch_size=6)
trainer.train(load_best_model=False)
