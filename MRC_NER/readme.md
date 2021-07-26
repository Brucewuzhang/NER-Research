# MRC NER

algorithm in the paper [A Unified MRC Framework for Named Entity Recognition](https://aclanthology.org/2020.acl-main.519.pdf)


training command: 
    python -m MRC_NER.train --data_dir ../NER/conll2003 --model_dir ../NER/MRC_model --max_len 256 --batch_size 16
    
    
    
Result:

* direct finetuning the whole model: micro precision: 0.8899, micro recall： 0.9024, micro f1: 0.8961

* two stage training: micro precision: 0.8977, micro recall： 0.9043, micro f1: 0.9010