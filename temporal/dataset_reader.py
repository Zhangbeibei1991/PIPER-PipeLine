import os
import json
import pickle
from collections import OrderedDict
from typing import Dict, List, Tuple

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, SpanField, AdjacencyField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data import Token


@DatasetReader.register('temporal')
class TemporalDatasetReader(DatasetReader):
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            max_tokens: int = -1,  # `130` may be appropriate
            collect_adj: bool = False,
            triplet: bool = False,
            flip: bool = False,
            **kwargs
    ):

        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tok_seq": SingleIdTokenIndexer(namespace='token_vocab'),
                                                  "pos_seq": SingleIdTokenIndexer(namespace='pos_tag_vocab',
                                                                                  feature_name='pos_'),
                                                  "ent_seq": SingleIdTokenIndexer(namespace='ent_tag_vocab',
                                                                                  feature_name='ent_type_'),
                                                  "tag_seq": SingleIdTokenIndexer(namespace='tag_tag_vocab',
                                                                                  feature_name='tag_')}
        self._max_tokens = max_tokens
        self._collect_adj = collect_adj
        self._triplet = triplet
        self._flip = flip
        self._rev_map = OrderedDict([('VAGUE', 'VAGUE'),
                                     ('BEFORE', 'AFTER'),
                                     ('AFTER', 'BEFORE'),
                                     ('SIMULTANEOUS', 'SIMULTANEOUS'),
                                     ('INCLUDES', 'IS_INCLUDED'),
                                     ('IS_INCLUDED', 'INCLUDES')])

    def _parse(self, item):
        words = item['token']
        postags = item['pos']
        ents = ['O'] * len(words)
        tenses = ['O'] * len(words)
        e1_span = item['e1_span']
        e2_span = item['e2_span']
        label = item['label']
        if label == "EQUAL":
            label = "SIMULTANEOUS"
        meta = [item['doc_id'] + '_' + _ for _ in item['event_pairs_id']]

        for idx in range(e1_span[0], e1_span[1]):
            ents[idx] = item.get('e1_type', 'O')
            tenses[idx] = item.get('e1_tense', 'O')

        for idx in range(e2_span[0], e2_span[1]):
            ents[idx] = item.get('e2_type', 'O')
            tenses[idx] = item.get('e2_tense', 'O')

        tokens = []
        for i in range(len(words)):
            tokens.append(Token(
                text=words[i],
                pos_=postags[i],
                ent_type_=ents[i],
                tag_=tenses[i]
            ))

        if self._max_tokens > 0:
            tokens = tokens[:self._max_tokens]
            assert e1_span[1] <= self._max_tokens and e2_span[1] <= self._max_tokens, \
                f"The `max_tokens` is set to {self._max_tokens}, " \
                f"but the events are not included at index: {max(e1_span[1], e2_span[1])}."

        # adj matrix
        dep_rels = []
        dep_word_idx = []
        for rel, widx in item['dep']:
            if rel == 'ROOT':
                rel = 'root'
            dep_rels.append(rel)
            dep_word_idx.append(widx)

        adj_indices = []
        adj_labels = []
        for j, dep_relation in enumerate(dep_rels):
            if j >= len(words):
                break
            token1_id, token2_id = j, int(dep_word_idx[j])
            if token2_id == -1 or token2_id >= len(words):
                token2_id = token1_id

            if self._max_tokens > 0:
                if token1_id < self._max_tokens and token2_id < self._max_tokens:
                    adj_indices.append((token1_id, token2_id))
                    adj_labels.append(dep_relation)
                    if token1_id != token2_id:
                        adj_indices.append((token2_id, token1_id))
                        adj_labels.append(dep_relation)
            else:
                adj_indices.append((token1_id, token2_id))
                adj_labels.append(dep_relation)
                if token1_id != token2_id:
                    adj_indices.append((token2_id, token1_id))
                    adj_labels.append(dep_relation)

        if self._max_tokens > 0:
            for i in range(min(self._max_tokens, len(words))):
                if (i, i) not in adj_indices:
                    adj_indices.append((i, i))
                    adj_labels.append("self-loop")
        else:
            for i in range(len(words)):
                if (i, i) not in adj_indices:
                    adj_indices.append((i, i))
                    adj_labels.append("self-loop")

        return tokens, e1_span, e2_span, label, meta, (adj_indices, adj_labels)

    def _read(self, file_path: str):

        # load json items
        with open(file_path, 'r') as f:
            items = json.load(f)

        if self._triplet:
            docid_to_items = {}
            for item in items:
                if item['doc_id'] not in docid_to_items.keys():
                    docid_to_items[item['doc_id']] = [item]
                else:
                    docid_to_items[item['doc_id']].append(item)

            # combination
            for doc_items in docid_to_items.values():
                already_set = []
                for x in doc_items:
                    for y in doc_items:
                        for z in doc_items:
                            if x['event_pairs_id'][1] == y['event_pairs_id'][0] and \
                                    x['event_pairs_id'][0] == z['event_pairs_id'][0] and \
                                    y['event_pairs_id'][1] == z['event_pairs_id'][1] and \
                                    [x['event_pairs_id'], y['event_pairs_id'], z['event_pairs_id']] not in already_set:
                                already_set.append([x['event_pairs_id'], y['event_pairs_id'], z['event_pairs_id']])

                                x_tokens, x_e1_span, x_e2_span, x_label, x_meta, x_adj = self._parse(x)
                                y_tokens, y_e1_span, y_e2_span, y_label, y_meta, y_adj = self._parse(y)
                                z_tokens, z_e1_span, z_e2_span, z_label, z_meta, z_adj = self._parse(z)

                                yield self.text_to_instance(
                                    sent1_tokens=x_tokens,
                                    sent1_e1_span=x_e1_span,
                                    sent1_e2_span=x_e2_span,
                                    sent1_label=x_label,
                                    sent1_meta=x_meta,
                                    sent1_adj=x_adj,

                                    sent2_tokens=y_tokens,
                                    sent2_e1_span=y_e1_span,
                                    sent2_e2_span=y_e2_span,
                                    sent2_label=y_label,
                                    sent2_meta=y_meta,
                                    sent2_adj=y_adj,

                                    sent3_tokens=z_tokens,
                                    sent3_e1_span=z_e1_span,
                                    sent3_e2_span=z_e2_span,
                                    sent3_label=z_label,
                                    sent3_meta=z_meta,
                                    sent3_adj=z_adj
                                )

        else:
            for item in items:
                tokens, e1_span, e2_span, label, meta, adj = self._parse(item)

                if self._flip:
                    rev_e1_span = e2_span
                    rev_e2_span = e1_span
                    rev_label = self._rev_map[label]
                    rev_meta = list(reversed(meta))
                    yield self.text_to_instance(
                        sent1_tokens=tokens,
                        sent1_e1_span=e1_span,
                        sent1_e2_span=e2_span,
                        sent1_label=label,
                        sent1_meta=meta,
                        sent1_adj=adj,

                        sent2_tokens=tokens,
                        sent2_e1_span=rev_e1_span,
                        sent2_e2_span=rev_e2_span,
                        sent2_label=rev_label,
                        sent2_meta=rev_meta,
                        sent2_adj=adj
                    )
                else:
                    yield self.text_to_instance(
                        sent1_tokens=tokens,
                        sent1_e1_span=e1_span,
                        sent1_e2_span=e2_span,
                        sent1_label=label,
                        sent1_meta=meta,
                        sent1_adj=adj,
                    )

    def text_to_instance(self,
                         sent1_tokens: List[Token],
                         sent1_e1_span: List[int],
                         sent1_e2_span: List[int],
                         sent1_label: str,
                         sent1_meta: List[str],
                         sent1_adj: Tuple[List, List] = None,
                         sent2_tokens: List[Token] = None,
                         sent2_e1_span: List[int] = None,
                         sent2_e2_span: List[int] = None,
                         sent2_label: str = None,
                         sent2_meta: List[str] = None,
                         sent2_adj: Tuple[List, List] = None,
                         sent3_tokens: List[Token] = None,
                         sent3_e1_span: List[int] = None,
                         sent3_e2_span: List[int] = None,
                         sent3_label: str = None,
                         sent3_meta: List[str] = None,
                         sent3_adj: Tuple[List, List] = None, ):

        sequence1 = TextField(sent1_tokens, self._token_indexers)
        fields: Dict[str, Field] = {
            "sent1_tokens": sequence1,
            "sent1_e1_span": SpanField(sent1_e1_span[0], sent1_e1_span[1] - 1, sequence1),
            "sent1_e2_span": SpanField(sent1_e2_span[0], sent1_e2_span[1] - 1, sequence1),
            "sent1_label": LabelField(sent1_label),
            "sent1_meta": MetadataField({"pairs_idx": sent1_meta})
        }
        if self._collect_adj:
            fields.update({
                "sent1_adj": AdjacencyField(indices=sent1_adj[0], sequence_field=sequence1, labels=sent1_adj[1],
                                            padding_value=0, label_namespace='adj_tag_vocab')
            })

        if sent2_tokens is not None:
            sequence2 = TextField(sent2_tokens, self._token_indexers)
            fields.update({
                "sent2_tokens": sequence2,
                "sent2_e1_span": SpanField(sent2_e1_span[0], sent2_e1_span[1] - 1, sequence2),
                "sent2_e2_span": SpanField(sent2_e2_span[0], sent2_e2_span[1] - 1, sequence2),
                "sent2_label": LabelField(sent2_label),
                "sent2_meta": MetadataField({"pairs_idx": sent2_meta})
            })
            if self._collect_adj:
                fields.update({
                    "sent2_adj": AdjacencyField(indices=sent2_adj[0], sequence_field=sequence2, labels=sent2_adj[1],
                                                padding_value=0, label_namespace='adj_tag_vocab')
                })

        if sent3_tokens is not None:
            sequence3 = TextField(sent3_tokens, self._token_indexers)
            fields.update({
                "sent3_tokens": sequence3,
                "sent3_e1_span": SpanField(sent3_e1_span[0], sent3_e1_span[1] - 1, sequence3),
                "sent3_e2_span": SpanField(sent3_e2_span[0], sent3_e2_span[1] - 1, sequence3),
                "sent3_label": LabelField(sent3_label),
                "sent3_meta": MetadataField({"pairs_idx": sent3_meta})
            })
            if self._collect_adj:
                fields.update({
                    "sent3_adj": AdjacencyField(indices=sent3_adj[0], sequence_field=sequence3, labels=sent3_adj[1],
                                                padding_value=0, label_namespace='adj_tag_vocab')
                })

        return Instance(fields)


if __name__ == "__main__":
    from allennlp.common.util import ensure_list
    from allennlp.data import PyTorchDataLoader, AllennlpDataset
    from allennlp.data.vocabulary import Vocabulary

    reader = TemporalDatasetReader(max_tokens=130, collect_adj=True, triplet=True, flip=False)
    instances = ensure_list(reader.read('../data/matres_qiangning/test.json'))

    dataset = AllennlpDataset(instances)
    vocab = Vocabulary.from_instances(dataset)
    dataset.index_with(vocab)
    dataloader = PyTorchDataLoader(dataset, batch_size=4)
    for batch in dataloader:
        print(batch.keys())
