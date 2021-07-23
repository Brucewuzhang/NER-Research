import tensorflow as tf
from transformers import BertTokenizer

query = {"ORG": "organization entities are limited to named corporate, governmental, or other organizational entities.",
         "PER": "person entities are named persons or family.",
         "LOC": "location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
         "MISC": "examples of miscellaneous entities include events, nationalities, products and works of art."}


def encode_file(datafile, bert_version='bert-base-uncased', max_len=512):
    """generate data for flat ner e.g. conll2003, nested ner needs revision"""
    tokenizer = BertTokenizer.from_pretrained(bert_version)

    query_len = {}
    for k, v in query.items():
        query_len[k] = len(tokenizer(v)['input_ids'])

    end_token_id = tokenizer._convert_token_to_id('[SEP]')

    def gen():
        with open(datafile, 'r', encoding='utf-8') as f:
            text = f.read()

        seqs = text.split('\n\n')
        seqs = [seq.split('\n') for seq in seqs]
        for seq in seqs:
            # print(seq)
            if seq[0].startswith('-DOCSTART-'):
                continue

            labels = {}
            for k, _ in query_len.items():
                labels[k] = {'start': [],
                             'end': []}

            # todo: get start and end
            entries = [e.split(' ') for e in seq if e]
            ws = [e[0] for e in entries]
            ts = [e[-1] for e in entries]
            label_masks = []
            n_words = len(ws)
            curr_idx = 0
            for i, (w, t) in enumerate(zip(ws, ts)):
                tokens = tokenizer.tokenize(w)
                l = len(tokens)
                # we use the first subtoken to label its corresponding word, other subtokens are ignored when computing loss
                # they are not ignored for computing the contextual representation
                if t == 'O':
                    pass
                else:
                    b_or_i, cat = t.split('-')
                    if i == 0:
                        labels[cat]['start'].append(curr_idx)
                        if n_words > 1:
                            if ts[i+1] == 'O':
                                labels[cat]['end'].append(curr_idx)
                            else:
                                b_or_i_next, cat_next = ts[i+1].split('-')
                                if b_or_i_next == 'B' or cat != cat_next:
                                    labels[cat]['end'].append(curr_idx)
                        else:
                            # n_words == 1
                            labels[cat]['end'].append(curr_idx)

                    elif i == n_words - 1:
                        labels[cat]['end'].append(curr_idx)
                        if ts[i-1] == 'O':
                            labels[cat]['start'].append(curr_idx)
                        else:
                            b_or_i_prev, cat_prev = ts[i - 1].split('-')
                            if b_or_i_prev == 'B' or cat != cat_prev:
                                labels[cat]['start'].append(curr_idx)
                    else:
                        # n_words >=3

                        if ts[i - 1] == 'O':
                            labels[cat]['start'].append(curr_idx)
                        else:
                            b_or_i_prev, cat_prev = ts[i - 1].split('-')
                            if b_or_i == 'B' or cat != cat_prev:
                                # start case, B tag or proceeded by O tag
                                labels[cat]['start'].append(curr_idx)

                        if ts[i + 1] == 'O':
                            labels[cat]['end'].append(curr_idx)
                        else:
                            b_or_i_next, cat_next = ts[i + 1].split('-')
                            if b_or_i_next == 'B' or cat != cat_next:
                                # end case, followed by 'O' for 'B' tag
                                labels[cat]['end'].append(curr_idx)

                label_masks.extend([1] + [0] * (len(tokens) - 1))
                curr_idx += l

            for k, v in labels.items():
                for k2, v2 in v.items():
                    v[k2] = [i + query_len[k] for i in v2]

                # query len + context len + 1 (eos len)
                v['label_masks'] = [0] * query_len[k] + label_masks + [0]

            if label_masks:
                for k, v in labels.items():
                    assert len(v['start']) == len(v['end'])
                    inputs = tokenizer(query[k], " ".join(ws))
                    inputs['label_masks'] = v['label_masks']
                    seq_len = len(v['label_masks'])
                    assert seq_len == len(inputs['input_ids'])
                    inputs['start_labels'] = [0] * seq_len
                    for i in v['start']:
                        inputs['start_labels'][i] = 1

                    inputs['end_labels'] = [0] * seq_len
                    for i in v['end']:
                        inputs['end_labels'][i] = 1

                    if seq_len > max_len:
                        # truncate to max length
                        seq_len = max_len
                        for k2, v2 in inputs.items():
                            inputs[k2] = v2[:seq_len]

                        # if max length is big than length of the longest query, we only need to change input ids and
                        # labels
                        inputs['input_ids'][-1] = end_token_id
                        inputs['start_labels'][-1] = 0
                        inputs['end_labels'][-1] = 0
                        inputs['label_masks'][-1] = 0

                    match_labels = [[0 for _ in range(seq_len)] for _ in range(seq_len)]
                    for s, e in zip(v['start'], v['end']):
                        if s < seq_len and e < seq_len:
                            match_labels[s][e] = 1

                    inputs['match_labels'] = match_labels

                    yield dict(inputs)

    return gen


def generate_dataset(datafile, bert_version='bert-base-uncased', batch_size=32,
                     shuffle=True, max_len=512):
    encoded_seq = encode_file(datafile, bert_version=bert_version, max_len=max_len)
    data_shapes = {'input_ids': [None],
                   'token_type_ids': [None],
                   'attention_mask': [None],
                   "start_labels": [None],
                   "end_labels": [None],
                   "match_labels": [None, None],
                   'label_masks': [None]}
    data_types = {'input_ids': tf.int32,
                  'token_type_ids': tf.int32,
                  'attention_mask': tf.int32,
                  "start_labels": tf.int32,
                  "end_labels": tf.int32,
                  "match_labels": tf.int32,
                  'label_masks': tf.bool}

    dataset = tf.data.Dataset.from_generator(encoded_seq, output_shapes=data_shapes,
                                             output_types=data_types)
    dataset = dataset.padded_batch(batch_size, padded_shapes=data_shapes)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=50_000, reshuffle_each_iteration=True)
    dataset = dataset
    return dataset


if __name__ == '__main__':
    datafile = '/data/share/user/bruce.zhang/NER/conll2003/eng.train'
    # datafile = '/data/share/user/bruce.zhang/NER/conll2003/eng.testb'

    # gen = encode_file(datafile)
    # for example in gen():
    #     print(example)
    #     break

    dataset = generate_dataset(datafile, batch_size=32, shuffle=False)
    for e in dataset.take(-1):
        print(e)
