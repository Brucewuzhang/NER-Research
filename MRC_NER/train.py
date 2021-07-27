import argparse
import os
import tensorflow as tf
from .utils import generate_dataset
from .model import MRCNER, MRCBiLSTMNER
from bert_ner.optimization import create_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data dir contain train/val/test files', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--dropout_rate', type=float, help='drop out rate', default=0.1)
    parser.add_argument('--match_dropout_rate', type=float, help='span drop out rate', default=0.3)
    parser.add_argument('--start_loss_weight', type=float, help='start weight loss', default=1.0)
    parser.add_argument('--end_loss_weight', type=float, help='end weight loss', default=1.0)
    parser.add_argument('--match_loss_weight', type=float, help='match loss weight', default=0.1)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-5)
    parser.add_argument('--lr1', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--model_dir', help='model dir', required=True)
    parser.add_argument('--epoch', type=int, help='train epoch', default=3)
    parser.add_argument('--version', type=str, help='bert version', default='bert-base-uncased')
    parser.add_argument('--truecase', action='store_true', help='whether to do truecase', default=False)
    parser.add_argument('--bilstm', action='store_true', help='whether to use BertFeatureExtractionNER',
                        default=False)
    parser.add_argument('--max_len', type=int, help='whether to do truecase', default=512)

    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    batch_size = int(args.batch_size)
    dropout_rate = args.dropout_rate
    match_dropout_rate = args.match_dropout_rate
    start_loss_weight = args.start_loss_weight
    end_loss_weight = args.end_loss_weight
    match_loss_weight = args.match_loss_weight
    bert_version = args.version
    truecase = args.truecase
    max_len = args.max_len
    bilstm = args.bilstm
    print("Using bilstm to construct span feature? {}".format(bilstm))

    print("bert version: {}".format(bert_version))
    lr = args.lr
    lr1 = args.lr1
    epoch = int(args.epoch)
    os.makedirs(model_dir, exist_ok=True)

    train_file = os.path.join(data_dir, 'eng.train')
    val_file = os.path.join(data_dir, 'eng.testa')
    test_file = os.path.join(data_dir, 'eng.testb')

    train_dataset = generate_dataset(train_file, bert_version=bert_version, batch_size=batch_size, max_len=max_len)
    val_dataset = generate_dataset(val_file, bert_version=bert_version, batch_size=batch_size,
                                   shuffle=False, max_len=10_000)

    test_dataset = generate_dataset(test_file, bert_version=bert_version, batch_size=batch_size,
                                    shuffle=False, max_len=10_000)

    # create model
    filepath = os.path.join(model_dir, 'model.ckpt-{epoch}')
    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True,
                                              save_weights_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=1, verbose=1)

    if bilstm:
        model = MRCBiLSTMNER(dropout_rate=dropout_rate, match_dropout_rate=match_dropout_rate, initializer_range=0.02,
                             bert_version=bert_version, start_loss_weight=start_loss_weight,
                             end_loss_weight=end_loss_weight, match_loss_weight=match_loss_weight)
    else:
        model = MRCNER(dropout_rate=dropout_rate, match_dropout_rate=match_dropout_rate, initializer_range=0.02,
                       bert_version=bert_version, start_loss_weight=start_loss_weight,
                       end_loss_weight=end_loss_weight, match_loss_weight=match_loss_weight)

    # first stage only train dense layer
    model.bert.trainable = False
    model.bert_finetune = False

    for it in train_dataset.take(1):
        # print(i, t)
        _ = model(it)
        print(model.summary())
        # print(tf.keras.utils.plot_model(model))

    opt1 = tf.keras.optimizers.Adam(lr1)

    n_batches = 0
    for _ in train_dataset.take(-1):
        n_batches += 1

    step_per_epoch = int(n_batches * 1.01)
    warmup_steps = int(0.1 * step_per_epoch) * epoch

    opt2 = create_optimizer(init_lr=lr,
                            num_train_steps=step_per_epoch * epoch,
                            num_warmup_steps=warmup_steps,
                            optimizer_type='adamw')

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=opt1)

    if 'albert' not in bert_version:
        stage1_epoch = 10
        model.fit(train_dataset, epochs=stage1_epoch, validation_data=val_dataset, validation_freq=1, verbose=1,
                  callbacks=[early_stop])

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

    tp_all = 0
    fp_all = 0
    fn_all = 0
    for example in test_dataset:
        match_labels = example.pop('match_labels')
        match_logits = model(example, training=False)
        match_pred = match_logits > 0
        match_labels = tf.cast(match_labels, tf.bool)

        tp = tf.math.logical_and(match_labels, match_pred)
        tp = tf.reduce_sum(tf.cast(tp, dtype=tf.float32)).numpy()

        fp = tf.math.logical_and(tf.math.logical_not(match_labels), match_pred)
        fp = tf.reduce_sum(tf.cast(fp, dtype=tf.float32)).numpy()

        fn = tf.math.logical_and(match_labels, tf.math.logical_not(match_pred))
        fn = tf.reduce_sum(tf.cast(fn, dtype=tf.float32)).numpy()

        tp_all += tp
        fp_all += fp
        fn_all += fn

    precision = tp_all / (tp_all + fp_all + 1e-10)
    recall = tp_all / (tp_all + fn_all + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    print("micro precision: {:.4f}, micro recall： {:.4f}, micro f1: {:.4f}".format(precision, recall, f1))


if __name__ == "__main__":
    main()
