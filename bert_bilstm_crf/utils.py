import tensorflow as tf
from transformers import BertTokenizer, RobertaTokenizer
from seqeval.metrics import classification_report
import re


class NERF1Metrics(tf.keras.callbacks.Callback):

    def __init__(self, id2label, pad_value=0, validation_data=None):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
        """
        super(NERF1Metrics, self).__init__()
        self.id2label = id2label
        self.pad_value = pad_value
        self.validation_data = validation_data

    def score(self, y_true, y_pred):
        print(classification_report(y_true, y_pred, digits=4))

    def on_epoch_end(self, epoch, logs={}):
        self.on_epoch_end_fit(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        y_true = []
        y_pred = []
        for X in self.validation_data:
            tags = X.pop('tag')
            logits, seq_lens = self.model(X, training=False)
            for logit, seq_len, tag in zip(logits, seq_lens, tags.numpy()):
                viterbi_path = tf.argmax(logit, axis=-1)
                tag = tag[:seq_len]
                y_true.append([self.id2label[t] for t in tag])
                y_pred.append([self.id2label[t] for t in viterbi_path.numpy()])

        self.score(y_true, y_pred)


def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
    """A default set of length-bucket boundaries."""
    assert length_bucket_step > 1.0
    x = min_length
    boundaries = []
    while x < max_length:
        boundaries.append(x)
        x = max(x + 1, int(x * length_bucket_step))
    return boundaries


def batching_scheme(batch_size,
                    max_length,
                    min_length_bucket,
                    length_bucket_step,
                    drop_long_sequences=False,
                    shard_multiplier=1,
                    length_multiplier=1,
                    min_length=0):
    """A batching scheme based on model hyperparameters.
  Every batch contains a number of sequences divisible by `shard_multiplier`.
  Args:
    batch_size: int, total number of tokens in a batch.
    max_length: int, sequences longer than this will be skipped. Defaults to
      batch_size.
    min_length_bucket: int
    length_bucket_step: float greater than 1.0
    drop_long_sequences: bool, if True, then sequences longer than
      `max_length` are dropped.  This prevents generating batches with
      more than the usual number of tokens, which can cause out-of-memory
      errors.
    shard_multiplier: an integer increasing the batch_size to suit splitting
      across datashards.
    length_multiplier: an integer multiplier that is used to increase the
      batch sizes and sequence length tolerance.
    min_length: int, sequences shorter than this will be skipped.
  Returns:
     A dictionary with parameters that can be passed to input_pipeline:
       * boundaries: list of bucket boundaries
       * batch_sizes: list of batch sizes for each length bucket
       * max_length: int, maximum length of an example
  Raises:
    ValueError: If min_length > max_length
  """
    max_length = max_length or batch_size
    if max_length < min_length:
        raise ValueError("max_length must be greater or equal to min_length")

    boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                    length_bucket_step)
    boundaries = [boundary * length_multiplier for boundary in boundaries]
    max_length *= length_multiplier

    batch_sizes = [
        max(1, batch_size // length) for length in boundaries + [max_length]
    ]
    max_batch_size = max(batch_sizes)
    # Since the Datasets API only allows a single constant for window_size,
    # and it needs divide all bucket_batch_sizes, we pick a highly-composite
    # window size and then round down all batch sizes to divisors of that window
    # size, so that a window can always be divided evenly into batches.
    # TODO(noam): remove this when Dataset API improves.
    highly_composite_numbers = [
        1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
        2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
        83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
        720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
        7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
        36756720, 43243200, 61261200, 73513440, 110270160
    ]
    window_size = max(
        [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
    divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
    batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
    window_size *= shard_multiplier
    batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
    # The Datasets API splits one window into multiple batches, which
    # produces runs of many consecutive batches of the same size.  This
    # is bad for training.  To solve this, we will shuffle the batches
    # using a queue which must be several times as large as the maximum
    # number of batches per window.
    max_batches_per_window = window_size // min(batch_sizes)
    shuffle_queue_size = max_batches_per_window * 3

    ret = {
        "boundaries": boundaries,
        "batch_sizes": batch_sizes,
        "min_length": min_length,
        "max_length": (max_length if drop_long_sequences else 10 ** 9),
        "shuffle_queue_size": shuffle_queue_size,
    }
    return ret


def build_vocab(datafile):
    vocab = {'<pad>': 0, '<unknown>': 1}
    labels = {}
    with open(datafile, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries = line.split(' ')
            w = entries[0]
            t = entries[-1]
            _ = vocab.setdefault(w.lower(), len(vocab))
            _ = labels.setdefault(t, len(labels))

    return vocab, labels


def convert_all_captial(w):
    if re.search('^[A-Z]+$', w):
        return w[0] + w[1:].lower()
    else:
        return w


def encode_file(datafile, labels, bert_version='bert-base-uncased', do_truecase=False, max_len=None):
    if 'roberta' in bert_version:
        tokenizer = RobertaTokenizer.from_pretrained(bert_version)
        add_space = ' '
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_version)
        add_space = ''

    def gen():
        with open(datafile, 'r', encoding='utf-8') as f:
            text = f.read()

        O_label = labels['O']

        seqs = text.split('\n\n')
        seqs = [seq.split('\n') for seq in seqs]
        for seq in seqs:
            if seq[0].startswith('-DOCSTART-'):
                continue
            entries = [e.split(' ') for e in seq if e]
            ws = [e[0] for e in entries]
            t_idx = [labels[e[-1]] for e in entries]
            tags = []
            label_masks = []
            for i, (w, t) in enumerate(zip(ws, t_idx)):
                # w = convert_all_captial(w)
                if i != 0:
                    w = add_space + w
                tokens = tokenizer.tokenize(w)
                tags.extend([t] * len(tokens))
                label_masks.extend([1] + [0] * (len(tokens) - 1))
            if tags:
                # print(entries)
                tags = [O_label] + tags + [O_label]
                label_masks = [0] + label_masks + [0]
                inputs = tokenizer(" ".join(ws))
                if max_len is not None:
                    for k, v in inputs.items():
                        inputs[k] = v[:max_len]
                    tags = tags[:max_len]
                    label_masks = label_masks[:max_len]

                inputs['tag'] = tags
                inputs['label_masks'] = label_masks
                yield dict(inputs)

    return gen


def example_length(ex):
    return tf.shape(ex['input_ids'])[0]


def generate_dataset(datafile, labels, bert_version='bert-base-uncased', batch_size=32,
                     shuffle=True, do_truecase=False, max_len=None, dynamic_batch=None):
    encoded_seq = encode_file(datafile, labels, bert_version=bert_version, do_truecase=do_truecase, max_len=max_len)
    if 'roberta' in bert_version:
        padded_shapes = {'input_ids': [None],
                         'attention_mask': [None],
                         "tag": [None],
                         'label_masks': [None]}

        output_types = {'input_ids': tf.int32,
                        'attention_mask': tf.int32,
                        "tag": tf.int32,
                        'label_masks': tf.int32}
    else:
        padded_shapes = {"tag": [None], 'input_ids': [None],
                         'token_type_ids': [None],
                         'attention_mask': [None],
                         'label_masks': [None]}

        output_types = {'input_ids': tf.int32,
                        'token_type_ids': tf.int32,
                        'attention_mask': tf.int32,
                        "tag": tf.int32,
                        'label_masks': tf.int32}
    dataset = tf.data.Dataset.from_generator(encoded_seq, output_shapes=padded_shapes,
                                             output_types=output_types)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=50_000, reshuffle_each_iteration=True)

    if dynamic_batch is not None:
        batch_config = batching_scheme(dynamic_batch, max_length=max_len,
                                       min_length=3, length_bucket_step=1.1,
                                       min_length_bucket=3)

        batch_config["batch_sizes"] = [max(b, 32) for b in batch_config["batch_sizes"]]
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(example_length,
                                                                               batch_config["boundaries"],
                                                                               batch_config["batch_sizes"],
                                                                               padded_shapes=padded_shapes
                                                                               ))
    else:
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)

    return dataset


if __name__ == '__main__':
    datafile = '/data/share/user/bruce.zhang/NER/conll2003/eng.train'
    # datafile = '/data/share/user/bruce.zhang/NER/conll2003/eng.testb'
    vocab, labels = build_vocab(datafile)
    print(len(vocab))
    print(labels)
    dataset = generate_dataset(datafile, labels, batch_size=6)
    for e in dataset.take(1):
        print(e)
