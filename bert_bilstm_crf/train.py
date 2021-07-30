import argparse
import os
import json
import tensorflow as tf
from .utils import generate_dataset, build_vocab
from .model import BertBilstmCRF, RobertaBilstmCRF
from seqeval.metrics import classification_report
import tensorflow_addons as tfa
from bert_ner.optimization import create_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data dir contain train/val/test files')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--dynamic_batch_size', type=int, help='dynamic batch size by token', default=None)
    parser.add_argument('--dropout_rate', type=float, help='drop out rate', default=0.1)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-5)
    parser.add_argument('--lr1', type=float, help='learning rate of stage 1', default=1e-3)
    parser.add_argument('--model_dir', help='model dir')
    parser.add_argument('--epoch', type=int, help='train epoch', default=4)
    parser.add_argument('--version', type=str, help='bert version', default='bert-base-uncased')
    parser.add_argument('--truecase', action='store_true', help='whether to do truecase', default=False)
    parser.add_argument('--max_len', type=int, help='whether to do truecase', default=512)
    parser.add_argument('--hidden_size', type=int, help='lstm units number', default=32)
    parser.add_argument('--two_stage', action='store_true', help='whether to do two stage training', default=False)
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    batch_size = int(args.batch_size)
    dynamic_batch_size = args.dynamic_batch_size
    print("dynamic batch size is: {}".format(dynamic_batch_size))
    dropout_rate = args.dropout_rate
    bert_version = args.version
    truecase = args.truecase
    max_len = args.max_len
    hidden_size = args.hidden_size
    two_stage_training = args.two_stage
    print("bert version: {}".format(bert_version))
    lr = args.lr
    lr1 = args.lr1
    epoch = int(args.epoch)
    os.makedirs(model_dir, exist_ok=True)

    train_file = os.path.join(data_dir, 'eng.train')
    val_file = os.path.join(data_dir, 'eng.testa')
    test_file = os.path.join(data_dir, 'eng.testb')

    vocab_file = os.path.join(data_dir, 'vocab.json')
    label_file = os.path.join(data_dir, 'label.json')
    if os.path.exists(vocab_file) and os.path.exists(label_file):

        with open(label_file, 'r', encoding='utf-8') as f:
            label = json.load(f)

    else:
        vocab, label = build_vocab(train_file)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f)

        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(label, f)

    id2label = {v: k for k, v in label.items()}

    train_dataset = generate_dataset(train_file, label, bert_version=bert_version, batch_size=batch_size,
                                     do_truecase=truecase, max_len=max_len, dynamic_batch=dynamic_batch_size)
    val_dataset = generate_dataset(val_file, label, bert_version=bert_version, batch_size=batch_size * 2, shuffle=False,
                                   do_truecase=truecase, max_len=None, dynamic_batch=dynamic_batch_size)
    test_dataset = generate_dataset(test_file, label, bert_version=bert_version, batch_size=batch_size * 2,
                                    shuffle=False, do_truecase=truecase, max_len=None, dynamic_batch=dynamic_batch_size)

    # create model
    label_size = len(label)

    filepath = os.path.join(model_dir, 'model.ckpt-{epoch}')
    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True,
                                              save_weights_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=1, verbose=1)
    # f1_callback = NERF1Metrics(id2label, validation_data=val_dataset)

    if 'roberta' in bert_version:
        model = RobertaBilstmCRF(hidden_size, label_size, dropout_rate=dropout_rate, initializer_range=0.02,
                                 bert_version=bert_version)
    else:
        model = BertBilstmCRF(hidden_size, label_size, dropout_rate=dropout_rate, initializer_range=0.02,
                              bert_version=bert_version)

    # first stage only train dense layer
    model.bert.trainable = False
    model.bert_finetune = False

    for it in train_dataset.take(1):
        # print(i, t)
        _ = model(it)
        print(model.summary())
        # print(tf.keras.utils.plot_model(model))

    n_batches = 0
    for _ in train_dataset.take(-1):
        n_batches += 1

    print("number of batches is: {}".format(n_batches))
    step_per_epoch = int(n_batches * 1.1)
    warmup_steps = int(0.1 * step_per_epoch) * epoch

    train_dataset = train_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)

    if two_stage_training:
        opt1 = tf.keras.optimizers.Adam(lr1)

        model.compile(optimizer=opt1)
        if 'roberta' in bert_version or 'electra' in bert_version:
            stage1_epoch = 10
        else:
            stage1_epoch = 15
        history = model.fit(train_dataset, epochs=stage1_epoch, validation_data=val_dataset, validation_freq=1, verbose=1,
                            callbacks=[early_stop])

    opt2 = create_optimizer(init_lr=lr,
                            num_train_steps=step_per_epoch * epoch,
                            num_warmup_steps=warmup_steps,
                            optimizer_type='adamw')

    # now tuning the whole model
    model.bert.trainable = True
    model.bert_finetune = True
    model.compile(optimizer=opt2)

    model.fit(train_dataset, epochs=epoch, validation_data=val_dataset, validation_freq=1, verbose=1,
              callbacks=[ckpt, early_stop])

    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    model.load_weights(latest_ckpt).expect_partial()

    model.bert_finetune = False
    model.compile()

    # evaluate the model
    y_true = []
    y_pred = []
    for it in test_dataset.take(-1):
        tags = it.pop('tag')
        label_masks = it['label_masks']
        logits, seq_lens = model(it, training=False)
        for logit, seq_len, tag, label_mask in zip(logits, seq_lens, tags.numpy(), label_masks.numpy()):
            viterbi_path, _ = tfa.text.viterbi_decode(logit[:seq_len], model.transition_params)
            tag = [t for t, flag in zip(tag, label_mask) if flag]
            viterbi_path = viterbi_path[: seq_len]
            y_true.append([id2label[t] for t in tag])
            y_pred.append([id2label[t] for t in viterbi_path])

    print(classification_report(y_true, y_pred, digits=3))


if __name__ == "__main__":
    main()
