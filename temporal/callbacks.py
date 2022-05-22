import logging
import numpy as np
from typing import Dict, Any, List

from allennlp.training.trainer import TrainerCallback, GradientDescentTrainer
from allennlp.data.dataloader import TensorDict

from sklearn.metrics import f1_score, precision_score, recall_score

from .ILP import ILPSolver

logger = logging.getLogger(__name__)


@TrainerCallback.register("global_ilp_inference")
class GlobalILPCallback(TrainerCallback):
    def __init__(self):
        super(GlobalILPCallback, self).__init__()
        self._train_fwd_probs = []
        self._train_bwd_probs = []
        self._train_fwd_pairs = []
        self._train_bwd_pairs = []
        self._train_labels = []

        self._dev_fwd_probs = []
        self._dev_bwd_probs = []
        self._dev_fwd_pairs = []
        self._dev_bwd_pairs = []
        self._dev_labels = []

    def reset(self):
        logger.info("### Global ILP Inference ####  Reset the all holders.")
        self._train_fwd_probs.clear()
        self._train_bwd_probs.clear()
        self._train_fwd_pairs.clear()
        self._train_bwd_pairs.clear()
        self._train_labels.clear()

        self._dev_fwd_probs.clear()
        self._dev_bwd_probs.clear()
        self._dev_fwd_pairs.clear()
        self._dev_bwd_pairs.clear()
        self._dev_labels.clear()

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:
        """
        This callback hook is called after the end of each batch. This is equivalent to `BatchCallback`.
        """
        if epoch == -1:
            return

        if is_training:
            self._train_fwd_probs.append(batch_outputs[0]['ilp_fwd_probs'])
            self._train_fwd_pairs.extend(batch_outputs[0]['ilp_fwd_pairs'])
            if batch_outputs[0]['ilp_bwd_probs'] is not None:
                self._train_bwd_probs.append(batch_outputs[0]['ilp_bwd_probs'])
                self._train_bwd_pairs.extend(batch_outputs[0]['ilp_bwd_pairs'])
            self._train_labels.append(batch_outputs[0]['ilp_labels'])
        else:
            self._dev_fwd_probs.append(batch_outputs[0]['ilp_fwd_probs'])
            self._dev_fwd_pairs.extend(batch_outputs[0]['ilp_fwd_pairs'])
            if batch_outputs[0]['ilp_bwd_probs'] is not None:
                self._dev_bwd_probs.append(batch_outputs[0]['ilp_bwd_probs'])
                self._dev_bwd_pairs.extend(batch_outputs[0]['ilp_bwd_pairs'])
            self._dev_labels.append(batch_outputs[0]['ilp_labels'])

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        """
        This callback hook is called after the end of each epoch. This is equivalent to `EpochCallback`.
        """
        if epoch == -1:
            return

        if 'matres' in trainer.model.source:
            labels = [x for x in range(trainer.model.vocab.get_vocab_size(namespace=trainer.model.label_namespace))
                      if x != trainer.model.vocab.get_token_index('VAGUE', namespace=trainer.model.label_namespace)]
        elif 'tbd' in trainer.model.source:
            labels = [x for x in range(trainer.model.vocab.get_vocab_size(namespace=trainer.model.label_namespace))
                      if x != trainer.model.vocab.get_token_index('SIMULTANEOUS', namespace=trainer.model.label_namespace)]
        else:
            raise KeyError('The source {} is not compatible.'.format(trainer.model.source))

        label2idx = trainer.model.vocab.get_token_to_index_vocabulary(trainer.model.label_namespace)

        # calculate train
        if len(self._train_bwd_probs) > 0:
            logger.info("### Global ILP Inference ####  On training set, the solver will conduct in `flip` mode.")
            train_probs = np.concatenate(self._train_fwd_probs + self._train_bwd_probs)
            train_pairs = self._train_fwd_pairs + self._train_bwd_pairs
            train_flip = True
        else:
            logger.info("### Global ILP Inference ####  On training set, the solver will conduct in `normal` mode.")
            train_probs = np.concatenate(self._train_fwd_probs)
            train_pairs = self._train_fwd_pairs
            train_flip = False

        train_gold = np.concatenate(self._train_labels)
        train_pred = ILPSolver(train_pairs, train_probs, label2idx, flip=train_flip).inference()

        train_f1 = f1_score(y_true=train_gold, y_pred=train_pred, average='micro', labels=labels)
        train_r = recall_score(y_true=train_gold, y_pred=train_pred, average='micro', labels=labels)
        train_p = precision_score(y_true=train_gold, y_pred=train_pred, average='micro', labels=labels)

        logger.info(f"### Global ILP Inference ####  Train: F1: {train_f1}, P: {train_p}, R: {train_r}")

        # calculate dev
        if len(self._dev_bwd_probs) > 0:
            logger.info("### Global ILP Inference ####  On development set, the solver will conduct in `flip` mode.")
            dev_probs = np.concatenate(self._dev_fwd_probs + self._dev_bwd_probs)
            dev_pairs = self._dev_fwd_pairs + self._dev_bwd_pairs
            dev_flip = True
        else:
            logger.info("### Global ILP Inference ####  On development set, the solver will conduct in `normal` mode.")
            dev_probs = np.concatenate(self._dev_fwd_probs)
            dev_pairs = self._dev_fwd_pairs
            dev_flip = False

        dev_gold = np.concatenate(self._dev_labels)
        dev_pred = ILPSolver(dev_pairs, dev_probs, label2idx, flip=dev_flip).inference()[:dev_gold.shape[0]]

        dev_f1 = f1_score(y_true=dev_gold, y_pred=dev_pred, average='micro', labels=labels)
        dev_r = recall_score(y_true=dev_gold, y_pred=dev_pred, average='micro', labels=labels)
        dev_p = precision_score(y_true=dev_gold, y_pred=dev_pred, average='micro', labels=labels)

        logger.info(f"### Global ILP Inference ####  Dev: F1: {dev_f1}, P: {dev_p}, R: {dev_r}")

        # reset
        self.reset()

    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        """
        This callback hook is called after the final training epoch. The `epoch` is passed as an argument.
        """
        print()


