from transformers import TFBertModel, TFRobertaModel, TFElectraModel
import tensorflow as tf
import tensorflow_addons as tfa


class BertBilstmCRF(tf.keras.Model):
    def __init__(self, hidden_size, n_labels, dropout_rate=0.1, initializer_range=0.02,
                 bert_version='bert-base-uncased'):
        super(BertBilstmCRF, self).__init__()
        if 'bert' in bert_version:
            self.bert = TFBertModel.from_pretrained(bert_version)
        elif 'electra' in bert_version:
            self.bert = TFElectraModel.from_pretrained(bert_version)
        else:
            raise Exception("Not implemented Error for bert version: {}".format(bert_version))

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size // 2, return_sequences=True))
        self.transition_params = tf.Variable(tf.random.uniform((n_labels, n_labels)))
        self.dense = tf.keras.layers.Dense(n_labels, kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=initializer_range))
        self.bert_finetune = True

    def call(self, inputs, training=None, mask=None):
        tags = inputs.pop('tag', None)
        label_masks = inputs.pop('label_masks', None)
        # masks = inputs['attention_mask']

        # length of words
        text_lens = tf.math.reduce_sum(tf.cast(label_masks, tf.int32), axis=-1)
        # text_lens = tf.math.reduce_sum(masks, axis=-1)
        bert_output = self.bert(inputs, training=self.bert_finetune)

        drop_out = self.dropout(bert_output[0], training=training)
        lstm_out = self.bilstm(drop_out)
        lstm_out = self.dropout(lstm_out, training=training)
        logits = self.dense(lstm_out)

        # transform logits, moving all labeled tokens to the front of each sequence,
        # all labeled token logits will be moved to the front and other logits
        # will be put to the back
        cumsum = tf.math.cumsum(label_masks, axis=1)
        arg_mask = tf.cast(tf.math.equal(label_masks, 0), label_masks.dtype) * 10000
        sorted_idx = tf.argsort(cumsum + arg_mask)
        batch_size = tf.shape(label_masks)[0]
        l = tf.shape(label_masks)[1]
        batch_idx = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1)), (1, l))
        gather_idx = tf.stack([batch_idx, sorted_idx], axis=-1)
        logits = tf.gather_nd(logits, gather_idx)

        if tags is not None:
            tags = tf.gather_nd(tags, gather_idx)
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits,
                                                                                 tags,
                                                                                 text_lens,
                                                                                 self.transition_params)
            loss = -tf.math.reduce_mean(log_likelihood)
            self.add_loss(loss)

        return logits, text_lens


class RobertaBilstmCRF(tf.keras.Model):
    def __init__(self, hidden_size, n_labels, dropout_rate=0.1, initializer_range=0.02,
                 bert_version='roberta-base'):
        super(RobertaBilstmCRF, self).__init__()
        self.bert = TFRobertaModel.from_pretrained(bert_version)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size // 2, return_sequences=True))
        self.transition_params = tf.Variable(tf.random.uniform((n_labels, n_labels)))
        self.dense = tf.keras.layers.Dense(n_labels, kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=initializer_range))
        self.bert_finetune = True

    def call(self, inputs, training=None, mask=None):
        tags = inputs.pop('tag', None)
        label_masks = inputs.pop('label_masks', None)
        # masks = inputs['attention_mask']

        # length of words
        text_lens = tf.math.reduce_sum(tf.cast(label_masks, tf.int32), axis=-1)
        # text_lens = tf.math.reduce_sum(masks, axis=-1)
        bert_output = self.bert(inputs, training=self.bert_finetune)

        drop_out = self.dropout(bert_output[0], training=training)
        lstm_out = self.bilstm(drop_out)
        lstm_out = self.dropout(lstm_out, training=training)
        logits = self.dense(lstm_out)

        # transform logits, moving all labeled tokens to the front of each sequence,
        # all labeled token logits will be moved to the front and other logits
        # will be put to the back
        cumsum = tf.math.cumsum(label_masks, axis=1)
        arg_mask = tf.cast(tf.math.equal(label_masks, 0), label_masks.dtype) * 10000
        sorted_idx = tf.argsort(cumsum + arg_mask)
        batch_size = tf.shape(label_masks)[0]
        l = tf.shape(label_masks)[1]
        batch_idx = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1)), (1, l))
        gather_idx = tf.stack([batch_idx, sorted_idx], axis=-1)
        logits = tf.gather_nd(logits, gather_idx)

        if tags is not None:
            tags = tf.gather_nd(tags, gather_idx)
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits,
                                                                                 tags,
                                                                                 text_lens,
                                                                                 self.transition_params)
            loss = -tf.math.reduce_mean(log_likelihood)
            self.add_loss(loss)

        return logits, text_lens
