## A Unified Model for Joint Chinese Word Segmentation and Dependency Parsing

This is the code for the paper [A Unified Model for Joint Chinese Word Segmentation and Dependency Parsing](https://arxiv.org/abs/1904.04697)

#### Requirements
This project needs the natural language processing python package 
[fastNLP](https://github.com/fastnlp/fastNLP). You can install by
the following command

```bash
pip install fastNLP
```


### Data
Your data should in the format as following
```
1	中国	_	NR	NR	_	4	nn	_	_
2	残疾人	_	NN	NN	_	4	nn	_	_
3	体育	_	NN	NN	_	4	nn	_	_
4	事业	_	NN	NN	_	5	nsubj	_	_
5	方兴未艾	_	VV	VV	_	0	root	_	_

1	新华社	_	NR	NR	_	12	dep	_	_
```
The 1st, 3rd, 6th, 7th(starts from 0) column should be words, pos tags,
 dependency heads and dependency labels, respectively. Empty line separate
  two instances.

You should place your data like the following structure
```
-JointCwsParser
    ...
    -train.py
    -train_bert.py
-data
    -ctb5
        -train.conll
        -dev.conll
        -test.conll
    -ctb7
        -...
    -ctb9
        -...
```
We use code from https://github.com/hankcs/TreebankPreprocessing to convert the original format into the conll format.


### Run the code
You can directly run by
```
python train.py --dataset ctb5
```
or 
```
python train_bert.py --dataset ctb5
```
FastNLP will download pretrained embeddings or BERT weight automatically.
 
