import argparse
import os
import json
import tensorflow as tf
from .utils import generate_dataset, build_vocab, NERF1Metrics
from .model import BiLstmCRF
from seqeval.metrics import classification_report
import tensorflow_addons as tfa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data dir contain train/val/test files')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--dropout_rate', type=float, help='drop out rate', default=0.3)
    parser.add_argument('--embedding_size', type=int, help='embedding size', default=32)
    parser.add_argument('--hidden_dim', type=int, help='lstm units', default=32)
    parser.add_argument('--lr', help='learning rate', default=1e-3)
    parser.add_argument('--model_dir', help='model dir')
    parser.add_argument('--epoch', type=int, help='train epoch', default=50)

    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    batch_size = int(args.batch_size)
    dropout_rate = args.dropout_rate
    embedding_size = args.embedding_size
    hidden_dim = args.hidden_dim
    lr = args.lr
    epoch = int(args.epoch)
    os.makedirs(model_dir, exist_ok=True)

    train_file = os.path.join(data_dir, 'eng.train')
    val_file = os.path.join(data_dir, 'eng.testa')
    test_file = os.path.join(data_dir, 'eng.testb')

    vocab_file = os.path.join(data_dir, 'vocab.json')
    label_file = os.path.join(data_dir, 'label.json')
    if os.path.exists(vocab_file) and os.path.exists(label_file):
        print("vocab file and label file exist, direct loading...")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        with open(label_file, 'r', encoding='utf-8') as f:
            label = json.load(f)

    else:
        vocab, label = build_vocab(train_file)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f)

        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(label, f)

    id2label = {v: k for k,v in label.items()}

    train_dataset = generate_dataset(train_file, vocab, label, batch_size)
    val_dataset = generate_dataset(val_file, vocab, label, batch_size, shuffle=False)
    test_dataset = generate_dataset(test_file, vocab, label, batch_size, shuffle=False, testset=False)

    # create model
    vocab_size = len(vocab)
    label_size = len(label)

    filepath = os.path.join(model_dir, 'model.ckpt-{epoch}')
    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True,
                                              save_weights_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
    # f1_callback = NERF1Metrics(id2label, validation_data=val_dataset)

    internal_model = BiLstmCRF(vocab_size, embedding_size, hidden_dim, label_size, dropout_rate)
    # inp = tf.keras.layers.Input(shape=[None], name='input')
    # tag = tf.keras.layers.Input(shape=[None], name='tags')
    # potentials, seq_lens, log_likelihood = internal_model([inp, tag], training=True)
    # model = tf.keras.Model([inp, tag], potentials)
    # loss = -tf.math.reduce_mean(log_likelihood)
    # model.add_loss(loss)

    model = internal_model

    for it in train_dataset.take(1):
        # print(i, t)
        _ = model(it[0])
        print(model.summary())
        # print(tf.keras.utils.plot_model(model))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    model.fit(train_dataset, epochs=epoch, validation_data=val_dataset, validation_freq=1, verbose=2,
              callbacks=[ckpt, early_stop])

    latest_ckpt = tf.train.latest_checkpoint(model_dir)

    model.load_weights(latest_ckpt).expect_partial()

    y_true = []
    y_pred = []
    for it in test_dataset.take(-1):
        it = it[0]
        tags = it[1]
        logits, seq_lens = model(inputs=(it[0],), training=False)
        for logit, seq_len, tag in zip(logits, seq_lens, tags.numpy()):
            viterbi_path, _ = tfa.text.viterbi_decode(logit[:seq_len], model.transition_params)
            tag = tag[:seq_len]
            y_true.append([id2label[t] for t in tag])
            y_pred.append([id2label[t] for t in viterbi_path])

    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()




