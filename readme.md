# My repo to study different NER architecture and to implement them.

Very easy to understand and read-friendly implementations. Some training may need different learning rate for
the transfered layers and newly added layers, we take a two stage training strategy instead of a single pass training
with two different optimizers.

**If you have any questions regarding my implementations or suggestions, please file an issue.**


## Requirements:
* Tensorflow 2.2 and other python packages
* python >=3.6

Create the conda env I used:

    conda env create -f conda_env.yml

## My current best result on conll2003: f1: 91.9

## Algorithms implemented:
* Machine Reading Comprehension based NER classification
* Bert|RoBerta + bilstm +crf
* Bert|RoBerta + dense
* bilstm +crf


### Roberta + bilstm layer

    python -m bert_bilstm_crf.train --data_dir ../NER/conll2003 --model_dir ../NER/BertBiLstm --version roberta-base --two_stage --epoch 4

| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
|ORG    |  0.885  |  0.926   |  0.905  |    1661|
 |    MISC  |    0.783  |   0.813   |  0.798   |    702|
 |     PER   |   0.975   |  0.960   |  0.968   |   1617|
 |     LOC   |   0.939   |  0.937   |  0.938    |  1668|
|micro avg  |    0.913  |   0.925  |   0.919    |  5648|
|macro avg   |   0.914   |  0.925   |  0.919   |   5648|


### Roberta + dense layer

| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
   |   ORG   |   0.882   |  0.910   |  0.896   |   1661|
   |   LOC   |   0.928   |  0.933   |  0.931   |   1668|
  |    PER   |   0.962  |   0.970  |   0.966  |    1617|
  |   MISC   |   0.768  |   0.823  |   0.795 |      702|
|micro avg  |    0.903  |   0.924  |   0.913  |    5648|
|macro avg  |    0.904  |   0.924  |   0.914  |    5648|


### Bert + dense layer

    python -m bert_ner.train --data_dir /path/to/data --model_dir /path/to/saved/model --epoch 3

results on conll2003 dataset

| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
|   PER    |   0.97    |  0.96   |   0.97  |    1617|
|   ORG     |  0.85 |     0.89 |     0.87   |   1661|
| MISC     |  0.77  |    0.78 |     0.78  |     702|
|  LOC   |    0.92  |    0.93  |    0.92 |     1668|
|micro avg |      0.90  |    0.91   |   0.90  |    5648|
|macro avg  |     0.90  |    0.91  |    0.90 |     5648|

### Bert + Bi-LSTM CRF

    python -m bert_bilstm_crf.train --data_dir /path/to/data --model_dir /path/to/save/model --epoch 3 --hidden_size 32 --two_stage
or 

    python -m bert_bilstm_crf.train --data_dir /path/to/data --model_dir /path/to/save/model --epoch 3 --hidden_size 32 --two_stage --dynamic_batch_size 1800

results on conll2003 dataset

| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
|MISC   |    0.79   |   0.77  |    0.78  |     702|
 |     PER  |     0.97  |    0.97 |     0.97 |     1617|
 |     LOC  |     0.91  |    0.94 |     0.92 |     1668|
 |     ORG  |     0.89   |   0.88  |    0.89 |     1661|
|micro avg  |     0.91   |   0.91  |    0.91  |    5648|
|macro avg  |     0.91   |   0.91  |    0.91  |    5648|


### MRC NER: roberta-base + bilstm span feature

* micro precision: micro precision: 0.9159, micro recall： 0.9105, micro f1: 0.9132



### bilstm crf
    
    python -m bilstm_crf.train --data_dir /path/to/data --model_dir /path/to/save/model --epoch 50

results on conll2003 dataset

|  Type|precision|recall|f1-score|support|
|:---:|---------|------|--------|-------|
|  LOC|0.67|0.69|0.68|1668|
|  MISC|       0.70  |    0.63 |     0.66   |    702|
 |     PER |      0.74  |    0.51    |  0.61 |     1617|
 |     ORG  |     0.64  |    0.58     | 0.61  |    1661 |
|micro avg  |     0.68  |    0.60   |   0.64  |    5648 |
|macro avg  |     0.68  |    0.60   |   0.64  |    5648 |


todo: Like reproted in BERT, use the document information in the training data?


