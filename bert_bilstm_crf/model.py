from transformers import TFBertModel
import tensorflow as tf
import tensorflow_addons as tfa


class BertBilstmCRF(tf.keras.Model):
    def __init__(self, hidden_size, n_labels, dropout_rate=0.1, initializer_range=0.02,
                 bert_version='bert-base-uncased'):
        super(BertBilstmCRF, self).__init__()
        self.bert = TFBertModel.from_pretrained(bert_version)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size // 2, return_sequences=True))
        self.transition_params = tf.Variable(tf.random.uniform((n_labels, n_labels)))
        self.dense = tf.keras.layers.Dense(n_labels, kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=initializer_range))

    def call(self, inputs, training=None, mask=None):
        tags = inputs.pop('tag', None)
        masks = inputs['attention_mask']
        text_lens = tf.math.reduce_sum(masks, axis=-1)
        bert_output = self.bert(inputs, training=training)

        drop_out = self.dropout(bert_output[0])
        lstm_out = self.bilstm(drop_out)
        logits = self.dense(lstm_out)
        if tags is not None:
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits,
                                                                                 tags,
                                                                                 text_lens,
                                                                                 self.transition_params)
            loss = tf.math.reduce_mean(log_likelihood)
            self.add_loss(loss)

        return logits, text_lens

