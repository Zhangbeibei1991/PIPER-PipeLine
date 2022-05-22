from typing import Dict, List, Any, TypeVar
from collections import OrderedDict
from overrides import overrides
import sklearn
import numpy as np
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator

from .classifiers import BasicTemporalClassifier, GCNTemporalClassifier, HDNTemporalClassifier
from .logic_losses import SymmetryLoss, ConjunctiveNot, ConjunctiveYes
from .conjunctive_table import tbd as tbd_conj
from .conjunctive_table import matres as matres_conj
from .metrics import ILPMetric

T = TypeVar("T", bound="FromParams")


@Model.register("logic_temporal")
class LogicTemporal(Model):
    classifiers = {
        'basic': BasicTemporalClassifier,
        'gcn': GCNTemporalClassifier,
        'hdn': HDNTemporalClassifier,
    }

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 pair_encoder: Seq2SeqEncoder,
                 dropout_rate: float = 0.,
                 gcn_num_layers: int = 0,
                 source: str = 'tbd',
                 ann_loss_wt: float = 1.,
                 sym_loss_wt: float = 1.,
                 conj_loss_wt: float = 1.,
                 classifier_type: str = 'basic',
                 label_namespace: str = "labels",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 verbose_metrics: bool = False,
                 ilp_inference: bool = False,
                 log_all_prf: bool = False,
                 plot_confusion_matrix: bool = False,
                 features: str = None,
                 **kwargs):
        super(LogicTemporal, self).__init__(vocab, **kwargs)
        assert source in ['tbd', 'matres', 'matres_qiangning']
        self.ann_loss_wt = ann_loss_wt
        self.sym_loss_wt = sym_loss_wt
        self.conj_loss_wt = conj_loss_wt

        self.source = source if '_' not in source else source.split('_')[0]
        self.label_namespace = label_namespace

        assert classifier_type in self.classifiers.keys()
        self.classifier = self.classifiers[classifier_type](
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            gcn_num_layers=gcn_num_layers,
            encoder=encoder,
            pair_encoder=pair_encoder,
            dropout_rate=dropout_rate,
            label_namespace=label_namespace,
            feature_list=features.split('#') if features else None
        )
        self._verbose_metrics = verbose_metrics

        self.acc = CategoricalAccuracy()

        if 'matres' in source:
            labels = [x for x in range(vocab.get_vocab_size(namespace=label_namespace))
                      if x != vocab.get_token_index('VAGUE', namespace=label_namespace)]
        elif 'tbd' in source:
            labels = [x for x in range(vocab.get_vocab_size(namespace=label_namespace))
                      if x != vocab.get_token_index('SIMULTANEOUS', namespace=label_namespace)]
        else:
            raise KeyError('The source {} is not compatible.'.format(source))

        self.f1 = FBetaMeasure(average='micro', beta=1., labels=labels)

        self.log_all_prf = log_all_prf
        if self.log_all_prf:
            self.all_f1 = FBetaMeasure()

        self._ilp_inference = ilp_inference
        if ilp_inference:
            self.ilp = ILPMetric(
                label2idx=vocab.get_token_to_index_vocabulary(label_namespace),
                labels=labels,
                flip=True  # default is flip mode, to solve sys constraints
            )

        self.plot_confusion_matrix = plot_confusion_matrix
        if self.plot_confusion_matrix:
            self.y_true = []
            self.y_pred = []

        self.rev_map = OrderedDict([('VAGUE', 'VAGUE'),
                                    ('BEFORE', 'AFTER'),
                                    ('AFTER', 'BEFORE'),
                                    ('SIMULTANEOUS', 'SIMULTANEOUS'),
                                    ('INCLUDES', 'IS_INCLUDED'),
                                    ('IS_INCLUDED', 'INCLUDES')])

        initializer(self)

    @overrides
    def forward(self,
                sent1_tokens: TextFieldTensors,
                sent1_e1_span: torch.LongTensor,
                sent1_e2_span: torch.LongTensor,
                sent1_label: torch.LongTensor,
                sent1_meta: List[Dict[str, Any]],
                sent1_adj=None,
                sent2_tokens=None,
                sent2_e1_span=None,
                sent2_e2_span=None,
                sent2_label=None,
                sent2_meta=None,
                sent2_adj=None,
                sent3_tokens=None,
                sent3_e1_span=None,
                sent3_e2_span=None,
                sent3_label=None,
                sent3_meta=None,
                sent3_adj=None
                ):

        mode = 'normal'
        if sent2_tokens is not None and sent3_tokens is None:
            mode = 'flip'
        if sent2_tokens is not None and sent3_tokens is not None:
            mode = 'triplet'

        if mode == 'normal':
            inputs = {
                "tokens": sent1_tokens,
                "e1_span": sent1_e1_span,
                "e2_span": sent1_e2_span,
                "label": sent1_label,
                "metadata": sent1_meta
            }
            if sent1_adj is not None:
                inputs["adj"] = sent1_adj

            outputs = self.classifier(**inputs)  # loss, probs, pairs, log_probs

            self.acc(outputs["logits"], inputs["label"])
            self.f1(outputs["logits"], inputs["label"])
            if self.log_all_prf:
                self.all_f1(outputs["logits"], inputs["label"])
            if self.plot_confusion_matrix:
                self.y_true.append(inputs["label"].detach().cpu().numpy())
                self.y_pred.append(outputs["logits"].argmax(-1).detach().cpu().numpy())

            if self._ilp_inference:
                self.ilp(
                    fwd_probs=outputs["probs"],
                    fwd_pairs=outputs["pairs"],
                    bwd_probs=None,
                    bwd_pairs=[],
                    labels=inputs["label"].detach().cpu().numpy()
                )

            return outputs

        if mode == 'flip':
            assert sent2_tokens is not None
            x_inputs = {
                "tokens": sent1_tokens,
                "e1_span": sent1_e1_span,
                "e2_span": sent1_e2_span,
                "label": sent1_label,
                "metadata": sent1_meta
            }
            y_inputs = {
                "tokens": sent2_tokens,
                "e1_span": sent2_e1_span,
                "e2_span": sent2_e2_span,
                "label": sent2_label,
                "metadata": sent2_meta
            }

            if sent1_adj is not None:
                x_inputs["adj"] = sent1_adj
                y_inputs["adj"] = sent2_adj

            outputs = {}
            x_outputs = self.classifier(**x_inputs)
            y_outputs = self.classifier(**y_inputs)
            alpha = x_outputs["log_probs"]
            beta = y_outputs["log_probs"]

            label2idx = self.vocab.get_token_to_index_vocabulary(self.label_namespace)
            sl = SymmetryLoss()

            sym_loss = torch.tensor(0, dtype=torch.float, requires_grad=False).to(x_inputs['label'].device)
            for label, idx in label2idx.items():
                alpha_label = label
                beta_label = self.rev_map[alpha_label]
                alpha_idx = idx
                beta_idx = label2idx[beta_label]
                sym_loss += sl(alpha, beta, alpha_idx, beta_idx).mean()
            sym_loss = sym_loss / len(label2idx)

            outputs["loss"] = self.ann_loss_wt * (x_outputs["loss"] + y_outputs["loss"]) + self.sym_loss_wt * sym_loss
            outputs["sym_loss"] = sym_loss.detach().cpu().item()

            self.acc(x_outputs["logits"], x_inputs["label"])  # just count the forward metric
            self.f1(x_outputs["logits"], x_inputs["label"])
            if self.log_all_prf:
                self.all_f1(x_outputs["logits"], x_inputs["label"])
            if self.plot_confusion_matrix:
                self.y_true.append(x_inputs["label"].detach().cpu().numpy())
                self.y_pred.append(x_outputs["logits"].argmax(-1).detach().cpu().numpy())

            probs = x_outputs["probs"]
            pairs = x_outputs["pairs"]
            rev_probs = y_outputs["probs"]
            rev_pairs = y_outputs["pairs"]

            if self._ilp_inference:
                self.ilp(
                    fwd_probs=probs,
                    fwd_pairs=pairs,
                    bwd_probs=rev_probs,
                    bwd_pairs=rev_pairs,
                    labels=x_inputs["label"].detach().cpu().numpy()
                )

            return outputs

        if mode == 'triplet':
            assert sent3_tokens is not None

            x_inputs = {
                "tokens": sent1_tokens,
                "e1_span": sent1_e1_span,
                "e2_span": sent1_e2_span,
                "label": sent1_label,
                "metadata": sent1_meta
            }
            y_inputs = {
                "tokens": sent2_tokens,
                "e1_span": sent2_e1_span,
                "e2_span": sent2_e2_span,
                "label": sent2_label,
                "metadata": sent2_meta
            }
            z_inputs = {
                "tokens": sent3_tokens,
                "e1_span": sent3_e1_span,
                "e2_span": sent3_e2_span,
                "label": sent3_label,
                "metadata": sent3_meta
            }

            if sent1_adj is not None:
                x_inputs["adj"] = sent1_adj
                y_inputs["adj"] = sent2_adj
                z_inputs["adj"] = sent3_adj

            outputs = {}
            x_outputs = self.classifier(**x_inputs)
            y_outputs = self.classifier(**y_inputs)
            z_outputs = self.classifier(**z_inputs)
            alpha = x_outputs["log_probs"]
            beta = y_outputs["log_probs"]
            gamma = z_outputs["log_probs"]

            cy = ConjunctiveYes(device=sent1_label.device)
            cn = ConjunctiveNot(device=sent1_label.device)

            label2idx = self.vocab.get_token_to_index_vocabulary(self.label_namespace)
            conj = tbd_conj if self.source == "tbd" else matres_conj

            conj_loss = torch.tensor(0, dtype=torch.float, requires_grad=False).to(x_inputs['label'].device)
            nx = 0
            for k, v in conj.items():
                alpha_label, beta_label = k
                alpha_idx, beta_idx = label2idx[alpha_label], label2idx[beta_label]

                # conj yes part
                for gamma_label in v['yes']:
                    nx += 1
                    gamma_idx = label2idx[gamma_label]
                    conj_loss += cy(alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx).mean()

                # conj no part
                # for gamma_label in v['not']:
                #     nx += 1
                #     gamma_idx = label2idx[gamma_label]
                #     conj_loss += cn(alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx).mean()
            conj_loss = conj_loss / nx

            outputs["loss"] = self.ann_loss_wt * (x_outputs["loss"] + y_outputs["loss"] + z_outputs["loss"]) + \
                              self.conj_loss_wt * conj_loss
            outputs["conj_loss"] = conj_loss

            # following codes are redundant: when evaluate, they are skipped
            # todo : delete
            self.acc(x_outputs["logits"], x_inputs["label"])
            self.f1(x_outputs["logits"], x_inputs["label"])
            if self.log_all_prf:
                self.all_f1(x_outputs["logits"], x_inputs["label"])
            if self.plot_confusion_matrix:
                self.y_true.append(x_inputs["label"].detach().cpu().numpy())
                self.y_pred.append(x_outputs["logits"].argmax(-1).detach().cpu().numpy())

            if self._ilp_inference:
                self.ilp(
                    fwd_probs=x_outputs["probs"],
                    fwd_pairs=x_outputs["pairs"],
                    bwd_probs=None,
                    bwd_pairs=[],
                    labels=x_inputs["label"].detach().cpu().numpy()
                )

            return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self.acc.get_metric(reset)
        }
        for k, v in self.f1.get_metric(reset).items():
            metrics.update({k: v})

        if self.log_all_prf:
            all_prf = self.all_f1.get_metric(reset)
            precision_list = all_prf["precision"]
            recall_list = all_prf["recall"]
            fscore_list = all_prf["fscore"]
            label2idx = self.vocab.get_token_to_index_vocabulary(namespace=self.label_namespace)
            for label, idx in label2idx.items():
                metrics.update({
                    label + "_p": precision_list[idx],
                    label + "_r": recall_list[idx],
                    label + "_f": fscore_list[idx]
                })

        if self._ilp_inference and reset:
            # we just inference at the end of epoch
            for k, v in self.ilp.get_metric(reset).items():
                metrics.update({k: v})

        if not self.training and self.plot_confusion_matrix and reset:
            self.y_pred = np.concatenate(self.y_pred)
            self.y_true = np.concatenate(self.y_true)
            labels = [x for x in self.vocab.get_token_to_index_vocabulary(self.label_namespace).keys()]
            cm = sklearn.metrics.confusion_matrix(y_pred=self.y_pred, y_true=self.y_true)
            self.plot_cm(cm, labels, show_number=False)
        return metrics

    @staticmethod
    def plot_cm(confusion_matrix, labels, save_path=None, show_number=True):
        import matplotlib.pyplot as plt
        plt.imshow(confusion_matrix, interpolation='none', cmap=plt.cm.binary)
        # plt.title('confusion_matrix')
        # plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation='vertical')
        plt.yticks(tick_marks, labels)

        if show_number:
            iters = np.reshape([[[i, j] for j in range(len(labels))] for i in range(len(labels))],
                               (confusion_matrix.size, 2))
            for i, j in iters:
                plt.text(j, i, format(confusion_matrix[i, j]))

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)