from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask

from .modules import GatedGCN, DistillModule


class BasicTemporalClassifier(nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 pair_encoder: Seq2SeqEncoder,
                 gcn_num_layers: int = 0,       # be consistent with the GCN type classifier
                 dropout_rate: float = 0.,
                 label_namespace: str = "labels"):
        super(BasicTemporalClassifier, self).__init__()
        self.vocab = vocab
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_classses = vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self.pair_encoder = pair_encoder
        self.linear1 = nn.Linear(encoder.get_output_dim() * 2 + pair_encoder.get_output_dim(), encoder.get_output_dim())
        self.linear2 = nn.Linear(encoder.get_output_dim(), self.num_classses)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                tokens: TextFieldTensors,
                e1_span: torch.LongTensor,  # (batch_size, 2)
                e2_span: torch.LongTensor,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None):
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        e1_hidden = []
        for i in range(batch_size):
            e1_hidden.append(encoded_text[i][e1_span[i][0]: e1_span[i][1] + 1].max(dim=0)[0])
        e1_hidden = torch.stack(e1_hidden, dim=0)

        e2_hidden = []
        for i in range(batch_size):
            e2_hidden.append(encoded_text[i][e2_span[i][0]: e2_span[i][1] + 1].max(dim=0)[0])
        e2_hidden = torch.stack(e2_hidden, dim=0)  # (batch_size, dim)

        features = list()
        features.append(e1_hidden * e2_hidden)  # commutative feat: times
        features.append(e1_hidden - e2_hidden)  # non-commutative feat: minus

        e1_e2_hidden = torch.stack([e1_hidden, e2_hidden], dim=1)  # (batch_size, 2, dim)
        e1_e2_encoded = self.pair_encoder(e1_e2_hidden, mask=None)  # (batch_size, 2, dim)
        features.append(e1_e2_encoded[:, -1, :])  # non-commutative feat: pair_encode

        features = torch.cat(features, dim=-1)
        out = self.dropout(self.linear1(features))
        logits = self.linear2(out)  # (batch_size, num_classes)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        pairs_key = []
        for each in metadata:
            pairs_key.append(tuple(each["pairs_idx"]))

        output_dict = {
            "logits": logits,
            "probs": probs.detach().cpu().numpy(),  # for ILPSolver
            "pairs": pairs_key,  # for ILPSolver
            "log_probs": log_probs,  # for consistency loss
        }

        if label is not None:
            loss = F.cross_entropy(logits, label.long().view(-1))
            output_dict["loss"] = loss

        return output_dict


class GCNTemporalClassifier(nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 pair_encoder: Seq2SeqEncoder,
                 gcn_num_layers: int = 0,
                 dropout_rate: float = 0.,
                 label_namespace: str = "labels"):
        super(GCNTemporalClassifier, self).__init__()
        self.vocab = vocab
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_classses = vocab.get_vocab_size(label_namespace)
        self.encoder = encoder

        # gcn
        self.gcn_layers = nn.ModuleList([GatedGCN(self.encoder.get_output_dim()) for _ in range(gcn_num_layers)])
        self.aggregate = nn.Linear(self.encoder.get_output_dim() * (gcn_num_layers + 1),
                                   self.encoder.get_output_dim())

        self.pair_encoder = pair_encoder
        self.linear1 = nn.Linear(encoder.get_output_dim() * 3 + pair_encoder.get_output_dim(), encoder.get_output_dim())
        self.linear2 = nn.Linear(encoder.get_output_dim(), self.num_classses)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                tokens: TextFieldTensors,
                e1_span: torch.LongTensor,  # (batch_size, 2)
                e2_span: torch.LongTensor,
                adj: torch.FloatTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None):
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        adj = adj.to(torch.bool).to(torch.float)
        gcn_inputs = encoded_text
        gcn_outputs = gcn_inputs
        layer_list = [gcn_inputs]

        for _, layer in enumerate(self.gcn_layers):
            gcn_outputs = layer(gcn_outputs, adj)
            gcn_outputs = self.dropout(gcn_outputs)
            layer_list.append(gcn_outputs)
        encoded_text = self.aggregate(torch.cat(layer_list, dim=-1))

        # reform the features
        e1_hidden = []
        for i in range(batch_size):
            e1_hidden.append(encoded_text[i][e1_span[i][0]: e1_span[i][1] + 1].max(dim=0)[0])
        e1_hidden = torch.stack(e1_hidden, dim=0)

        e2_hidden = []
        for i in range(batch_size):
            e2_hidden.append(encoded_text[i][e2_span[i][0]: e2_span[i][1] + 1].max(dim=0)[0])
        e2_hidden = torch.stack(e2_hidden, dim=0)  # (batch_size, dim)

        features = list()
        features.append(torch.cat([e1_hidden, e2_hidden], dim=-1))  # non-commutative feat: concat
        features.append(e1_hidden * e2_hidden)  # commutative feat: times
        features.append(e1_hidden - e2_hidden)  # non-commutative feat: minus

        e1_e2_hidden = torch.stack([e1_hidden, e2_hidden], dim=1)  # (batch_size, 2, dim)
        e1_e2_encoded = self.pair_encoder(e1_e2_hidden, mask=None)  # (batch_size, 2, dim)
        features.append(e1_e2_encoded[:, -1, :])  # non-commutative feat: pair_encode

        features = torch.cat(features, dim=-1)
        out = self.dropout(self.linear1(features))
        logits = self.linear2(out)  # (batch_size, num_classes)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        pairs_key = []
        for each in metadata:
            pairs_key.append(tuple(each["pairs_idx"]))

        output_dict = {
            "logits": logits,
            "probs": probs.detach().cpu().numpy(),  # for ILPSolver
            "pairs": pairs_key,  # for ILPSolver
            "log_probs": log_probs,  # for consistency loss
        }

        if label is not None:
            loss = F.cross_entropy(logits, label.long().view(-1))
            output_dict["loss"] = loss

        return output_dict


class HDNTemporalClassifier(nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 pair_encoder: Seq2SeqEncoder,
                 gcn_num_layers: int = 0,
                 dropout_rate: float = 0.,
                 label_namespace: str = "labels",
                 feature_list: List = None):
        super(HDNTemporalClassifier, self).__init__()
        self.vocab = vocab
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_classses = vocab.get_vocab_size(label_namespace)
        self.encoder = encoder

        # gcn
        self.num_layers = gcn_num_layers
        self.gcn_layers = nn.ModuleList([GatedGCN(self.encoder.get_output_dim()) for _ in range(gcn_num_layers)])
        self.dm_layers = nn.ModuleList([DistillModule(encoder.get_output_dim(), para_attention=False) for _ in range(gcn_num_layers)])
        self.aggregate = nn.Linear(self.encoder.get_output_dim() * (gcn_num_layers + 1),
                                   self.encoder.get_output_dim())

        self.feature_list = ["avg", "times", "pair_encode", "concat", "minus"] if feature_list is None else feature_list
        n_repeat = 0
        for x in self.feature_list:
            if x in ["avg", "times", "pair_encode", "minus"]:
                n_repeat += 1
            elif x in ["concat"]:
                n_repeat += 2
            else:
                raise ValueError("UnKnown feature !")
        if "pair_encode" in self.feature_list:
            self.pair_encoder = pair_encoder
        print("========= HDN features: [" + ' '.join(self.feature_list) + "] ===========")
        self.linear1 = nn.Linear(encoder.get_output_dim() * n_repeat, encoder.get_output_dim())
        self.linear2 = nn.Linear(encoder.get_output_dim(), self.num_classses)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                tokens: TextFieldTensors,
                e1_span: torch.LongTensor,  # (batch_size, 2)
                e2_span: torch.LongTensor,
                adj: torch.FloatTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None):
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        adj = adj.to(torch.bool).to(torch.float)
        gcn_inputs = encoded_text
        gcn_outputs = gcn_inputs
        layer_list = [gcn_inputs]

        for _, layer in enumerate(self.gcn_layers):
            gcn_outputs = layer(gcn_outputs, adj)
            gcn_outputs = self.dropout(gcn_outputs)
            layer_list.append(gcn_outputs)

        dm_mask = mask.to(torch.float)
        ns = torch.tensor(0, device=dm_mask.device)
        for i in range(self.num_layers):
            o1, o2, n1, n2 = self.dm_layers[i](layer_list[i], layer_list[i + 1], dm_mask, dm_mask)
            ns = ns + n1 + n2
            layer_list[i] = o1
            layer_list[i + 1] = o2
        ns = ns / (2 * self.num_layers)

        encoded_text = self.dropout(self.aggregate(torch.cat(layer_list, dim=-1)))

        # reform the features
        e1_hidden = []
        for i in range(batch_size):
            e1_hidden.append(encoded_text[i][e1_span[i][0]: e1_span[i][1] + 1].max(dim=0)[0])
        e1_hidden = torch.stack(e1_hidden, dim=0)

        e2_hidden = []
        for i in range(batch_size):
            e2_hidden.append(encoded_text[i][e2_span[i][0]: e2_span[i][1] + 1].max(dim=0)[0])
        e2_hidden = torch.stack(e2_hidden, dim=0)  # (batch_size, dim)

        features = list()

        if "times" in self.feature_list:
            features.append(e1_hidden * e2_hidden)  # commutative feat: times
        if "avg" in self.feature_list:
            features.append((e1_hidden + e2_hidden) / 2)  # commutative feat: avg

        if "pair_encode" in self.feature_list:
            e1_e2_hidden = torch.stack([e1_hidden, e2_hidden], dim=1)  # (batch_size, 2, dim)
            e1_e2_encoded = self.pair_encoder(e1_e2_hidden, mask=None)  # (batch_size, 2, dim)
            features.append(e1_e2_encoded[:, -1, :])  # non-commutative feat: pair_encode
        if "concat" in self.feature_list:
            features.append(torch.cat([e1_hidden, e2_hidden], dim=-1))  # non-commutative feat: concat
        if "minus" in self.feature_list:
            features.append(e1_hidden - e2_hidden)  # non-commutative feat: minus

        features = torch.cat(features, dim=-1)
        out = self.dropout(self.linear1(features))
        logits = self.linear2(out)  # (batch_size, num_classes)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        pairs_key = []
        for each in metadata:
            pairs_key.append(tuple(each["pairs_idx"]))

        output_dict = {
            "logits": logits,
            "probs": probs.detach().cpu().numpy(),  # for ILPSolver
            "pairs": pairs_key,  # for ILPSolver
            "log_probs": log_probs,  # for consistency loss
        }

        if label is not None:
            loss = F.cross_entropy(logits, label.long().view(-1)) + 0.05 * ns
            output_dict["loss"] = loss

        return output_dict
