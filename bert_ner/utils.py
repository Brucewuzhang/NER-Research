import tensorflow as tf
from transformers import BertTokenizer
from seqeval.metrics import classification_report


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


class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applys a warmup schedule on a given learning rate decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_schedule_fn,
      warmup_steps,
      power=1.0,
      name=None):
    super(WarmUpSchedule, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(global_step_float < warmup_steps_float,
                     lambda: warmup_learning_rate,
                     lambda: self.decay_schedule_fn(step),
                     name=name)


def build_vocab(datafile):
    vocab = {'<pad>': 0, '<unknown>': 1}
    labels = {}
    with open(datafile, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w, _, _, t = line.split(' ')
            _ = vocab.setdefault(w.lower(), len(vocab))
            _ = labels.setdefault(t, len(labels))

    return vocab, labels


def encode_file(datafile, labels, bert_version='bert-base-uncased'):
    def gen():
        with open(datafile, 'r', encoding='utf-8') as f:
            text = f.read()

        tokenizer = BertTokenizer.from_pretrained(bert_version)
        O_label = labels['O']

        seqs = text.split('\n\n')
        seqs = [seq.split('\n') for seq in seqs]
        for seq in seqs:
            if seq[0].startswith('-DOCSTART-'):
                continue
            entries = [e.split(' ') for e in seq if e]
            ws = [e[0] for e in entries]
            t_idx = [labels[e[3]] for e in entries]
            tags = []
            label_masks = []
            for w, t in zip(ws, t_idx):
                tokens = tokenizer.tokenize(w)
                tags.extend([t] * len(tokens))
                label_masks.extend([1] + [0] * (len(tokens) - 1))
            if tags:
                # print(entries)
                tags = [O_label] + tags + [O_label]
                label_masks = [0] + label_masks + [0]
                inputs = tokenizer(" ".join(ws))
                inputs['tag'] = tags
                inputs['label_masks'] = label_masks
                yield dict(inputs)

    return gen


def generate_dataset(datafile, labels, bert_version='bert-base-uncased', batch_size=32, shuffle=True):
    encoded_seq = encode_file(datafile, labels, bert_version=bert_version)
    dataset = tf.data.Dataset.from_generator(encoded_seq, output_shapes={'input_ids': [None],
                                                                         'token_type_ids': [None],
                                                                         'attention_mask': [None],
                                                                         "tag": [None],
                                                                         'label_masks': [None]},
                                             output_types={'input_ids': tf.int32,
                                                           'token_type_ids': tf.int32,
                                                           'attention_mask': tf.int32,
                                                           "tag": tf.int32,
                                                           'label_masks': tf.int32})
    dataset = dataset.padded_batch(batch_size, padded_shapes={"tag": [None], 'input_ids': [None],
                                                              'token_type_ids': [None],
                                                              'attention_mask': [None],
                                                              'label_masks': [None]})
    if shuffle:
        dataset = dataset.shuffle(buffer_size=200_000, reshuffle_each_iteration=True)
    dataset = dataset
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
