local seed = 42;
local cuda_device = 0;
local dataset = "tbd";
local num_epochs = 30;
local batch_size = 4;
local bert = "bert-base-uncased";
local bert_size = 768;
local warmup_steps=600;
local classifier_type = "hdn";
local hidden_size = 572;
local gcn_num_layers = 3;
local features = "avg#times#pair_encode#minus#concat";

# tuned
local dropout_rate = 0.6;
local conj_loss_wt = 1.0;
local lr = 5e-5;

local pretrained_path = "experiments_v2/flip_gcn_bert_optuna_tbd/trial_0";


{
    "numpy_seed": seed,
    "pytorch_seed": seed,
    "random_seed": seed,

    "vocabulary": {
        "type": "from_files",
        "directory": pretrained_path + "/vocabulary"
    },

    "dataset_reader" : {
        "type": "temporal",
        "token_indexers": {
            "tok_seq": {
                "type": "single_id",
                "namespace": "token_vocab",
            },
            "pos_seq": {
                "type": "single_id",
                "namespace": "pos_tag_vocab",
                "feature_name": "pos_",
            },
            "wpc_seq": {
                "type": "pretrained_transformer_mismatched",
                "model_name": bert,
                "namespace": "wordpiece_vocab",
            },
        },
        "triplet": true,
        "flip": false,
        "collect_adj": true,
        "max_tokens": 130,
    },

    "validation_dataset_reader" : {
        "type": "temporal",
        "token_indexers": {
            "tok_seq": {
                "type": "single_id",
                "namespace": "token_vocab",
            },
            "pos_seq": {
                "type": "single_id",
                "namespace": "pos_tag_vocab",
                "feature_name": "pos_",
            },
            "wpc_seq": {
                "type": "pretrained_transformer_mismatched",
                "model_name": bert,
                "namespace": "wordpiece_vocab",
            },
        },
        "triplet": false,
        "flip": true,
        "collect_adj": true,
        "max_tokens": 130,
    },

    "train_data_path": "data/" + dataset + "/train.json",

    "validation_data_path": "data/" + dataset + "/dev.json",

    "test_data_path": "data/" + dataset + "/test.json",

    "evaluate_on_test": true,

    "model": {
        "type": "logic_temporal",
        "text_field_embedder": {
            "token_embedders": {
                "tok_seq": {
                    "type": "embedding",
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                    "embedding_dim": 100,
                    "trainable": false,
                    "vocab_namespace": "token_vocab",
                },
                "pos_seq": {
                   "type": "embedding",
                   "embedding_dim": 50,
                   "trainable": true,
                   "vocab_namespace": "pos_tag_vocab",
                },
                "wpc_seq": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": bert,
                    "train_parameters": true,
                },
            },
        },

        "encoder": {
            "type": "lstm",
            "input_size": 150 + bert_size,
            "hidden_size": hidden_size,
            "num_layers": 1,
            "bidirectional": true,
        },

        gcn_num_layers: gcn_num_layers,

        "pair_encoder": {
            "type": "lstm",
            "input_size": hidden_size * 2,
            "hidden_size": hidden_size * 2,
            "num_layers": 1,
            "bidirectional": false,
        },

        "source": dataset,

        "dropout_rate": dropout_rate,

        "classifier_type": classifier_type,
        "conj_loss_wt": conj_loss_wt,

        "initializer": {
            "regexes": [
                [".*",
                    {
                        "type": "pretrained",
                        "weights_file_path": pretrained_path + "/best.th",
                        "parameter_name_overrides": {},
                    }
                ]
            ],
            "prevent_regexes": ["prevent_init_regex"],
        },
    },

    "data_loader": {
        "batch_size": batch_size,
        "shuffle": true,
    },

    "trainer": {
        "type": "gradient_descent",
        "optimizer": {
            "type": "huggingface_adamw",
            "correct_bias": false,
            "lr": lr,
            "eps": 1e-8,
            "weight_decay": 0.01,
            "parameter_groups": [
                [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps": warmup_steps,
        },
        "checkpointer": {
            "num_serialized_models_to_keep": 1,
        },
        "validation_metric": "+fscore",
        "num_epochs": num_epochs,
        "cuda_device": cuda_device,
        "patience": 2,
    }
}