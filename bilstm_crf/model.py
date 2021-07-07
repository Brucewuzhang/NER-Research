import tensorflow as tf
import tensorflow_addons as tfa


class BiLstmCRF(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size, label_size, dropout_rate):
        super(BiLstmCRF, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)))

    def call(self, inputs, training=None, mask=None):
        # inp = inputs['input']
        # tags = inputs.get('tag', None)
        if len(inputs) == 1:
            inp = inputs[0]
            tags = None
        elif len(inputs) == 2:
            inp, tags = inputs
        else:
            raise Exception("inputs must be a list of length 1 or 2")
        mask = tf.math.not_equal(inp, 0)
        seq_lens = tf.math.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=-1)

        embs = self.embedding(inp)
        embs = self.dropout(embs, training=training)

        lstm_out = self.bilstm(embs, mask=mask)
        potentials = self.dense(lstm_out)
        if tags is not None:
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(potentials,
                                                                                 tags, seq_lens,
                                                                                 self.transition_params)
            # normalize loss by seq length
            loss = -tf.math.reduce_mean(log_likelihood/ tf.cast(seq_lens, dtype=log_likelihood.dtype))
            self.add_loss(loss)

        return potentials, seq_lens

