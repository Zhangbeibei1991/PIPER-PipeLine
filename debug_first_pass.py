import json
import shutil
import sys
import os

from allennlp.commands import main

config_file = "./training_config/normal_gcn_bert_optuna1.jsonnet"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": 0}})

serialization_dir = "/tmp/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)
print(os.listdir())
# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "temporal",
    "-o", overrides,
]

main()