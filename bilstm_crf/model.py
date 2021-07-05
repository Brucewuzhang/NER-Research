import tensorflow as tf


class BiLstmCRF(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size, label_size, dropout_rate):
        super(BiLstmCRF, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.crf_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)))