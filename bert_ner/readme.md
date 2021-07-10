# Bert Ner


#### # result if only train dense layer

| Type| precision  |  recall | f1-score |  support|
|-----|------------|---------|----------|---------|
|     ORG   |    0.60    |  0.69   |   0.64   |   1237|
 |     LOC   |    0.76  |    0.81    |  0.78   |   1283|
 |     PER    |   0.84   |   0.85  |    0.85   |   1433|
 |    MISC     |  0.53   |   0.59   |   0.56   |    608|
|micro avg   |    0.70  |    0.76 |     0.73   |   4561|
|macro avg    |   0.71   |   0.76   |   0.73    |  4561|



#### # result of two staged training, first stage finetuning final dense layer, sencond
stage finetune the whole model

| Type| precision  |  recall | f1-score |  support|
|-----|------------|---------|----------|---------|
  |   MISC   |    0.70    |  0.64   |   0.67  |     608|
   |   LOC   |    0.90    |  0.90  |    0.90  |    1283|
   |   PER   |    0.89   |   0.87   |   0.88   |   1433|
  |    ORG   |    0.76   |   0.87  |    0.81  |   1237|
|micro avg  |     0.83   |   0.85   |   0.84  |    4561|
|macro avg  |     0.83   |   0.85  |    0.84   |   4561|


#### # result of direct finetuning (train for 3 epoch is enough)
| Type| precision  |  recall | f1-score |  support|
|-----|------------|---------|----------|---------|
   |   PER    |   0.97   |   0.97    |  0.97  |    1433|
   |   ORG    |   0.84    |  0.84     | 0.84  |    1237|
   |  MISC    |   0.69   |   0.73  |    0.71   |    608|
 |     LOC    |   0.88    |  0.93   |   0.90   |   1283|
|micro avg   |    0.87  |   0.89    |  0.88   |   4561|
|macro avg    |   0.87   |   0.89   |   0.88   |   4561|


