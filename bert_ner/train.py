import argparse
import os
import json
import tensorflow as tf
from .utils import generate_dataset, build_vocab, WarmUpSchedule, NERF1Metrics
from .model import BertNER
from seqeval.metrics import classification_report
import tensorflow_addons as tfa
from .optimization import create_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data dir contain train/val/test files')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--dropout_rate', type=float, help='drop out rate', default=0.1)
    parser.add_argument('--lr', help='learning rate', default=5e-5)
    parser.add_argument('--model_dir', help='model dir')
    parser.add_argument('--epoch', type=int, help='train epoch', default=40)
    parser.add_argument('--version', type=str, help='bert version', default='bert-base-uncased')
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    batch_size = int(args.batch_size)
    dropout_rate = args.dropout_rate
    bert_version = args.version
    lr = args.lr
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

    train_dataset = generate_dataset(train_file, label, bert_version=bert_version, batch_size=batch_size)
    val_dataset = generate_dataset(val_file, label, bert_version=bert_version, batch_size=batch_size * 2, shuffle=False)
    test_dataset = generate_dataset(test_file, label, bert_version=bert_version, batch_size=batch_size * 2,
                                    shuffle=False)

    # create model
    label_size = len(label)

    filepath = os.path.join(model_dir, 'model.ckpt-{epoch}')
    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True,
                                              save_weights_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
    # f1_callback = NERF1Metrics(id2label, validation_data=val_dataset)

    internal_model = BertNER(label_size, dropout_rate=dropout_rate, initializer_range=0.02,
                             bert_version='bert-base-uncased')

    model = internal_model
    # first stage only train dense layer
    model.bert.trainable = False
    model.bert_finetune = False

    for it in train_dataset.take(1):
        # print(i, t)
        _ = model(it)
        print(model.summary())
        # print(tf.keras.utils.plot_model(model))

    opt1 = tf.keras.optimizers.Adam(1e-3)

    step_per_epoch = 450
    warmup_steps = int(0.1 * step_per_epoch) * epoch

    opt2 = create_optimizer(init_lr=lr,
                                         num_train_steps=step_per_epoch * epoch,
                                         num_warmup_steps=warmup_steps,
                                         optimizer_type='adamw')

    train_dataset = train_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    #model.compile(optimizer=opt1)
    #model.fit(train_dataset, epochs=20, validation_data=val_dataset, validation_freq=1, verbose=1,
    #          callbacks=[early_stop])

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

    y_true = []
    y_pred = []
    for it in test_dataset.take(-1):
        tags = it.pop('tag')
        label_masks = it.pop('label_masks')
        logits, seq_lens = model(it, training=False)
        for logit, seq_len, tag, label_mask in zip(logits, seq_lens, tags.numpy(), label_masks.numpy()):
            viterbi_path = tf.argmax(logit, axis=-1)
            tag = tag[1:seq_len - 1]
            viterbi_path = viterbi_path.numpy()[1: seq_len - 1]
            y_true.append([id2label[t] for t, mask in zip(tag, label_mask) if mask])
            y_pred.append([id2label[t] for t, mask in zip(viterbi_path, label_mask) if mask])

    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
