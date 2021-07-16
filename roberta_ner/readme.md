# Roberta Ner


#### # result of two staged training, first stage finetuning final dense layer, sencond stage finetune the whole model

| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
   |   ORG   |   0.882   |  0.910   |  0.896   |   1661|
   |   LOC   |   0.928   |  0.933   |  0.931   |   1668|
  |    PER   |   0.962  |   0.970  |   0.966  |    1617|
  |   MISC   |   0.768  |   0.823  |   0.795 |      702|
|micro avg  |    0.903  |   0.924  |   0.913  |    5648|
|macro avg  |    0.904  |   0.924  |   0.914  |    5648|

#### # direct finetune for 3 epoch

| Type| precision  |  recall | f1-score |  support|
|:-----:|------------|---------|----------|---------|
 |     PER  |    0.973 |    0.963  |   0.968 |     1617|
 |     ORG |     0.870  |   0.909 |    0.889 |     1661|
 |     LOC  |    0.923 |    0.936 |    0.929 |     1668|
 |    MISC |     0.779 |    0.829  |   0.803 |      702|
|micro avg  |    0.902 |    0.922 |    0.912 |     5648|
|macro avg  |    0.904 |    0.922  |   0.913  |    5648|