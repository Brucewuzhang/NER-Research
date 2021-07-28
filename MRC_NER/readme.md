# MRC NER

algorithm in the paper [A Unified MRC Framework for Named Entity Recognition](https://aclanthology.org/2020.acl-main.519.pdf)
    
MRC based NER method has the potential to incorporated data from other data sources which only have overlapping labels.
    
Result using bert-base-uncased:

training command: 

    python -m MRC_NER.train --data_dir ../NER/conll2003 --model_dir ../NER/MRC_model --max_len 256 --batch_size 16

* direct finetuning the whole model: micro precision: 0.8899, micro recall： 0.9024, micro f1: 0.8961

* two stage training: micro precision: 0.8977, micro recall： 0.9043, micro f1: 0.9010


Result using roberta-base:

training command: 

    python -m MRC_NER.train --data_dir ../NER/conll2003 --model_dir ../NER/MRC_model --max_len 256 --batch_size 8 \
        --version roberta-base

* direct finetuning the whole model: micro precision: 0.8692, micro recall： 0.9100, micro f1: 0.8891

* two stage training: micro precision: micro precision: 0.8656, micro recall： 0.9159, micro f1: 0.8900


Result using roberta-base + lstm feature span:

training command: 

    python -m MRC_NER.train --data_dir ../NER/conll2003 --model_dir ../NER/MRC_model --max_len 256 --batch_size 16 \
        --version roberta-base --bilstm

* direct finetuning the whole model: micro precision: 0.9122, micro recall： 0.8926, micro f1: 0.9023

* two stage training: micro precision: micro precision: 0.9159, micro recall： 0.9105, micro f1: 0.9132



Result using albert-base-v2 lstm feature span:

training command: 

    python -m MRC_NER.train --data_dir ../NER/conll2003 --model_dir ../NER/MRC_model --max_len 256 --batch_size 32 \
        --version albert-base-v2 --bilstm

* direct finetuning the whole model: micro precision: 0.8933, micro recall： 0.8484, micro f1: 0.8703


Result using distlroberta-base + lstm feature span:

training command: 

    python -m MRC_NER.train --data_dir ../NER/conll2003 --model_dir ../NER/MRC_model --max_len 256 --batch_size 16 \
        --version distilroberta-base --bilstm

* two stage training: micro precision: 0.8818, micro recall： 0.9107, micro f1: 0.8960


