from transformers import TFRobertaModel
import tensorflow as tf


class RobertaNER(tf.keras.Model):
    def __init__(self, n_labels, dropout_rate=0.1, initializer_range=0.02,
                 bert_version='roberta-base'):
        super(RobertaNER, self).__init__()
        self.bert = TFRobertaModel.from_pretrained(bert_version)
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

