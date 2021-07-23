from transformers import TFBertModel, BertConfig
import tensorflow as tf


class MRCNER(tf.keras.Model):
    """
    paper: https://aclanthology.org/2020.acl-main.519.pdf
    """

    def __init__(self, dropout_rate=0.1, match_dropout_rate=0.3, initializer_range=0.02,
                 bert_version='bert-base-uncased', start_loss_weight=1.0, end_loss_weight=1.0, match_loss_weight=0.1):
        super(MRCNER, self).__init__()
        self.bert = TFBertModel.from_pretrained(bert_version)
        self.bert_config = BertConfig.from_pretrained(bert_version)
        self.hidden_size = self.bert_config.hidden_size

        # dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # start classification layer
        self.start_out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=initializer_range))
        # end classification layer
        self.end_out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=initializer_range))
        # match classification layer
        self.match_out = MatchClassifier(self.hidden_size * 2, match_dropout_rate, initializer_range)

        self.bert_finetune = True
        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        total_weight = start_loss_weight + end_loss_weight + match_loss_weight
        self.start_weight = start_loss_weight / total_weight
        self.end_weight = end_loss_weight / total_weight
        self.match_weight = match_loss_weight / total_weight

    def call(self, inputs, training=None, mask=None):
        start_labels = inputs.pop('start_labels', None)
        end_labels = inputs.pop('end_labels', None)
        match_labels = inputs.pop('match_labels', None)
        label_masks = inputs.pop('label_masks', None)

        bert_out = self.bert(inputs, training=self.bert_finetune)
        drop_out = self.dropout(bert_out[0], training=training)  # shape [batch_size, seq_len, hidden_size]

        start_logits = self.start_out(drop_out)
        end_logits = self.end_out(drop_out)

        # construct concatenated bert out
        seq_len = tf.shape(drop_out)[1]
        # row_expand = tf.expand_dims(drop_out, axis=2) # [batch_size, seq_len, 1, hidden_size]
        # row_expand = tf.tile(row_expand, [1, 1, seq_len, 1]) # [batch_size, seq_len, seq_len, hidden_size]
        row_expand = self.expand(drop_out, axis=2, seq_len=seq_len)

        # column_expand = tf.expand_dims(drop_out, axis=1) # [batch_size, 1, seq_len, hidden_size]
        # column_expand = tf.tile(column_expand, [1, seq_len, 1, 1]) # [batch_size, seq_len, seq_len, hidden_size]
        column_expand = self.expand(drop_out, axis=1, seq_len=seq_len)

        row_label_mask = self.expand(label_masks, axis=2, seq_len=seq_len)
        column_label_mask = self.expand(label_masks, axis=1, seq_len=seq_len)
        match_label_mask = tf.math.logical_and(row_label_mask, column_label_mask)
        match_label_mask = tf.linalg.band_part(match_label_mask, num_lower=0, num_upper=-1)  # start <= end

        match_feature = tf.concat([row_expand, column_expand], axis=-1)  # [batch_size, seq_len, seq_len, hidden_size]
        match_logits = self.match_out(match_feature)
        if match_labels is not None:
            # compute and add loss
            start_loss = self.loss_function(start_labels, start_logits)
            label_masks = tf.cast(label_masks, dtype=start_loss.dtype)
            n_tokens = tf.math.reduce_sum(label_masks)
            start_loss *= label_masks
            start_loss /= n_tokens

            self.add_loss(start_loss * self.start_weight)

            end_loss = self.loss_function(end_labels, end_logits)
            end_loss *= label_masks
            end_loss /= n_tokens

            self.add_loss(end_loss * self.end_weight)

            # predicted and gold match loss (pred and gold in the official implementation)
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            start_pred_expand = self.expand(start_preds, axis=-1, seq_len=seq_len)
            end_pred_expand = self.expand(end_preds, axis=1, seq_len=seq_len)

            preds_mask = tf.math.logical_and(start_pred_expand, end_pred_expand)

            start_labels_expand = self.expand(start_labels, axis=-1, seq_len=seq_len)
            end_labels_expand = self.expand(end_labels, axis=1, seq_len=seq_len)
            gold_mask = tf.math.logical_and(start_labels_expand, end_labels_expand)

            match_mask = tf.math.logical_and(match_label_mask, tf.math.logical_or(preds_mask, gold_mask))
            match_loss = self.loss_function(match_labels, match_logits)
            match_mask = tf.cast(match_mask, dtype=match_loss.dtype)
            match_loss /= (tf.math.reduce_sum(match_mask) + 1e-9)  # prevent division by zero
            self.add_loss(match_loss * self.match_weight)

        return start_logits, end_logits, match_logits

    @staticmethod
    def expand(tensor, axis, seq_len):
        tensor = tf.expand_dims(tensor, axis=axis)
        rank = tf.rank(tensor)
        tensor = tf.tile(tensor, [seq_len if i == axis else 1 for i in tf.range(rank)])
        return tensor


class MatchClassifier(tf.keras.Model):
    """
    dense layer for classifying (start, end) pairs

    Given the concatenated hidden state of a start and end token,
    determine whether they should be a span of NER
    """

    def __init__(self, hidden_size, dropout_rate, initializer_range):
        """

        :param hidden_size: bert hidden size * 2
        :param dropout_rate:
        :param initializer_range:
        """
        super(MatchClassifier, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size,
                                            activation='gelu',
                                            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                stddev=initializer_range))
        self.dense2 = tf.keras.layers.Dense(1,
                                            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                stddev=initializer_range))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None, mask=None):
        out1 = self.dense1(inputs)
        out2 = self.dropout(out1, training=training)
        out = self.dense2(out2)
        return out
