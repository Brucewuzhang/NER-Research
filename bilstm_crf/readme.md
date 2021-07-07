# A simple bilstm crf NER architecture

#### results on conll-2003 dataset
|  Type|precision|recall|f1-score|support|
|:---:|---------|------|--------|-------|
|  LOC|0.67|0.69|0.68|1668|
|  MISC|       0.70  |    0.63 |     0.66   |    702|
 |     PER |      0.74  |    0.51    |  0.61 |     1617|
 |     ORG  |     0.64  |    0.58     | 0.61  |    1661 |
|micro avg  |     0.68  |    0.60   |   0.64  |    5648 |
|macro avg  |     0.68  |    0.60   |   0.64  |    5648  |