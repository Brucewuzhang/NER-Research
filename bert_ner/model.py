from transformers import TFBertModel
import tensorflow as tf


class BertNER(tf.keras.Model):
    def __init__(self, n_labels, dropout_rate=0.1, initializer_range=0.02,
                 bert_version='bert-base-uncased'):
        super(BertNER, self).__init__()
        self.bert = TFBertModel.from_pretrained(bert_version)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(n_labels, kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=initializer_range))
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.bert_finetune = True

    def call(self, inputs, training=None, mask=None):
        tags = inputs.pop('tag', None)
        label_masks = inputs.pop('label_masks', None)
        masks = inputs['attention_mask']
        text_lens = tf.math.reduce_sum(tf.cast(masks, tf.int32), axis=-1)
        bert_output = self.bert(inputs, training=self.bert_finetune)

        drop_out = self.dropout(bert_output[0], training=training)
        logits = self.dense(drop_out)
        if tags is not None:
            loss = self.loss_function(tags, logits)
            masks = tf.cast(masks, dtype=loss.dtype)
            masks *= tf.cast(label_masks, loss.dtype)
            loss *= masks
            loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(masks)
            self.add_loss(loss)

        return logits, text_lens


class BertFeatureExtractionNER(tf.keras.Model):
    def __init__(self, n_labels, dropout_rate=0.1, initializer_range=0.02,
                 bert_version='bert-base-uncased', lstm_units=768//2):
        super(BertFeatureExtractionNER, self).__init__()
        self.bert = TFBertModel.from_pretrained(bert_version, output_hidden_states=True)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        self.dense = tf.keras.layers.Dense(n_labels, kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=initializer_range))
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.bert_finetune = True

    def call(self, inputs, training=None, mask=None):
        tags = inputs.pop('tag', None)
        label_masks = inputs.pop('label_masks', None)
        masks = inputs['attention_mask']
        text_lens = tf.math.reduce_sum(tf.cast(masks, tf.int32), axis=-1)
        # for feature extraction model, bert is not trainable
        bert_output = self.bert(inputs, training=False)
        layer_outputs = bert_output[2]
        concat_context = tf.concat([layer_outputs[i] for i in range(-4, 0)], axis=-1)

        drop_out = self.dropout(concat_context, training=training)
        lstm_out1 = self.lstm1(drop_out)
        drop_out2 = self.dropout(lstm_out1, training=training)
        lstm_out2 = self.lstm2(drop_out2)
        logits = self.dense(lstm_out2)
        if tags is not None:
            loss = self.loss_function(tags, logits)
            masks = tf.cast(masks, dtype=loss.dtype)
            masks *= tf.cast(label_masks, loss.dtype)
            loss *= masks
            loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(masks)
            self.add_loss(loss)

        return logits, text_lens
