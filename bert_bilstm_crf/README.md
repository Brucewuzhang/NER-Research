# Bert BiLstm CRF NER


#### # result if only train the bisltm and dense layer (lstm units 32)

| Type| precision  |  recall | f1-score |  support|
|-----|------------|---------|----------|---------|
      |LOC    |   0.89    |  0.91    |  0.90    |  1668|
 |     PER    |   0.95    |  0.95    |  0.95  |    1617|
   |   ORG   |    0.87   |   0.84     | 0.85    |  1661|
  |   MISC  |     0.74   |   0.70    |  0.72   |    702|
|micro avg     |  0.88   |   0.88    |  0.88    |  5648|
|macro avg   |    0.88   |   0.88   |   0.88    |  5648|



#### # result of two staged training, first stage finetuning only bilstm crf, sencond stage finetune the whole model

| Type| precision  |  recall | f1-score |  support|
|-----|------------|---------|----------|---------|
|MISC   |    0.79   |   0.77  |    0.78  |     702|
 |     PER  |     0.97  |    0.97 |     0.97 |     1617|
 |     LOC  |     0.91  |    0.94 |     0.92 |     1668|
 |     ORG  |     0.89   |   0.88  |    0.89 |     1661|
|micro avg  |     0.91   |   0.91  |    0.91  |    5648|
|macro avg  |     0.91   |   0.91  |    0.91  |    5648|


#### # direct finetuning is very slow, or setting two optimizer with different learning rate is needed, so two stage training is prefered



