import tensorflow as tf
from transformers import RobertaTokenizer
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


def encode_file(datafile, labels, bert_version='roberta-base', do_truecase=False, max_len=None):
    def gen():
        with open(datafile, 'r', encoding='utf-8') as f:
            text = f.read()

        tokenizer = RobertaTokenizer.from_pretrained(bert_version)
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
                if i != 0:
                    w = ' ' + w
                tokens = tokenizer.tokenize(w)
                # we use the first subtoken to label its corresponding word, other subtokens are ignored when computing loss
                # they are not ignored for computing the contextual representation
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


def generate_dataset(datafile, labels, bert_version='bert-base-uncased', batch_size=32,
                     shuffle=True, do_truecase=False, max_len=None):
    encoded_seq = encode_file(datafile, labels, bert_version=bert_version, do_truecase=do_truecase, max_len=max_len)
    dataset = tf.data.Dataset.from_generator(encoded_seq, output_shapes={'input_ids': [None],
                                                                         'attention_mask': [None],
                                                                         "tag": [None],
                                                                         'label_masks': [None]},
                                             output_types={'input_ids': tf.int32,
                                                           'attention_mask': tf.int32,
                                                           "tag": tf.int32,
                                                           'label_masks': tf.int32})
    dataset = dataset.padded_batch(batch_size, padded_shapes={"tag": [None], 'input_ids': [None],
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
