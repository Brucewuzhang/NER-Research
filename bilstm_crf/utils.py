import tensorflow as tf


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


def encode_file(datafile, vocab: dict, labels: dict):
    def gen():
        with open(datafile, 'r', encoding='utf-8') as f:
            text = f.read()

        seqs = text.split('\n\n')
        seqs = [seq.split('\n') for seq in seqs]
        seqs.sort(key=len)
        for seq in seqs:
            entries = [e.split(' ') for e in seq if e]
            w_idx = [vocab.get(e[0].lower(), 1) for e in entries]
            t_idx = [labels[e[3]] for e in entries]
            if w_idx:
                # yield {"input": w_idx, "tag": t_idx}
                yield w_idx, t_idx
    return gen


def generate_dataset(datafile, vocab, labels, batch_size, shuffle=True):
    encoded_seq = encode_file(datafile, vocab, labels)
    dataset = tf.data.Dataset.from_generator(encoded_seq, output_shapes=([None], [None]), output_types=(tf.int32, tf.int32))
                                             # output_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),
                                             #                   tf.TensorSpec(shape=[None], dtype=tf.int32)))
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None]))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=200_000, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x,y: ((x, y),), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    datafile = '/data/share/user/bruce.zhang/NER/conll2003/eng.train'
    # datafile = '/data/share/user/bruce.zhang/NER/conll2003/eng.testb'
    vocab, labels = build_vocab(datafile)
    print(len(vocab))
    print(labels)
    dataset = generate_dataset(datafile, vocab, labels, batch_size=6)
    for e in dataset.take(1):
        print(e)

