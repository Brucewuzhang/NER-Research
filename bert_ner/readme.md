# Bert Ner


#### # result if only train dense layer

| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
 |     ORG     |  0.69   |   0.80   |   0.74   |   1661|
 |     PER   |    0.94    |  0.92   |   0.93   |   1617|
  |    LOC    |   0.84  |    0.85  |    0.85   |   1668|
 |    MISC   |    0.66   |   0.58  |    0.62   |    702|
|micro avg   |    0.80    |  0.82    |  0.81   |   5648|
|macro avg    |   0.80   |   0.82  |    0.81   |   5648|



#### # result of two staged training, first stage finetuning final dense layer, sencond stage finetune the whole model

| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
|   PER    |   0.97    |  0.96   |   0.97  |    1617|
|   ORG     |  0.85 |     0.89 |     0.87   |   1661|
| MISC     |  0.77  |    0.78 |     0.78  |     702|
|  LOC   |    0.92  |    0.93  |    0.92 |     1668|
|micro avg |      0.90  |    0.91   |   0.90  |    5648|
|macro avg  |     0.90  |    0.91  |    0.90 |     5648|


#### # result of direct finetuning
| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
 |     LOC   |    0.90   |   0.93   |   0.91  |    1668|
 |    MISC   |    0.77   |   0.75   |   0.76   |    702|
  |    ORG   |    0.85   |   0.88   |   0.86   |   1661|
  |    PER    |   0.97    |  0.97   |   0.97   |   1617|
|micro avg   |    0.89   |   0.90   |   0.89    |  5648|
|macro avg    |   0.89   |   0.90   |   0.89    |  5648|

