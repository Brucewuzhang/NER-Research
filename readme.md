# My repo to study different NER architecture and to implement them.

Very easy to understand and read-friendly implementations. Some training may need different learning rate for
the transfered layers and newly added layers, we take a two stage training strategy instead of a single pass training
with two different optimizers.


## Requirements:
* Tensorflow 2 and other python packages
* with a GeForce RTX 2080 Ti gpu, each training takes less than 20 minutes



Algorithms implemented:
* bilstm crf
    
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

* Bert + dense layer

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

* Bert + Bi-LSTM CRF

    python -m bert_bilstm_crf.train --data_dir /path/to/data --model_dir /path/to/save/model --epoch 3 --hidden_size 32 --two_stage
        

results on conll2003 dataset

| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
|MISC   |    0.79   |   0.77  |    0.78  |     702|
 |     PER  |     0.97  |    0.97 |     0.97 |     1617|
 |     LOC  |     0.91  |    0.94 |     0.92 |     1668|
 |     ORG  |     0.89   |   0.88  |    0.89 |     1661|
|micro avg  |     0.91   |   0.91  |    0.91  |    5648|
|macro avg  |     0.91   |   0.91  |    0.91  |    5648|

