import torch
from typing import Dict

from overrides import overrides
from allennlp.training.trainer import Trainer,GradientDescentTrainer
from allennlp.data.dataloader import TensorDict
import allennlp.nn.util as nn_util


@Trainer.register('my_gradient_descent')
class MyTrainer(GradientDescentTrainer):

    def __init__(self, **kwargs):
        super(MyTrainer, self).__init__(**kwargs)

    @overrides
    def batch_outputs(self, batch: TensorDict, for_training: bool) -> Dict[str, torch.Tensor]:
        """
                Does a forward pass on the given batch and returns the output dictionary that the model
                returns, after adding any specified regularization penalty to the loss (if training).
                """
        batch = nn_util.move_to_device(batch, self.cuda_device)
        output_dict = self._pytorch_model(**batch)

        if for_training:
            try:
                assert "loss" in output_dict
                regularization_penalty = self.model.get_regularization_penalty()

                if regularization_penalty is not None:
                    output_dict["reg_loss"] = regularization_penalty
                    output_dict["loss"] += regularization_penalty

            except AssertionError:
                if for_training:
                    raise RuntimeError(
                        "The model you are trying to optimize does not contain a"
                        " 'loss' key in the output of model.forward(inputs)."
                    )

        return output_dict


if __name__ == "__main__":
    MyTrainer()