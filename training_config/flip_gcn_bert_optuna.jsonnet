local seed = 42;
local cuda_device = 0;
local dataset = std.extVar('dataset');
local num_epochs = std.parseInt(std.extVar('flip_epochs'));
local batch_size = std.parseInt(std.extVar('flip_bsz'));
local bert = std.extVar("bert");
local bert_size = std.parseInt(std.extVar('bert_size'));
local warmup_steps=600;
local classifier_type = std.extVar("classifier_type");
local hidden_size = std.parseInt(std.extVar('hidden_size'));
local gcn_num_layers = std.parseInt(std.extVar('gcn_num_layers'));

# tuned
local dropout_rate = std.parseJson(std.extVar('dropout_rate'));
local sym_loss_wt = std.parseJson(std.extVar('sym_loss_wt'));
local lr = std.parseJson(std.extVar('flip_lr'));

local pretrained_path = std.extVar("flip_pretrained_path");


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
        "triplet": false,
        "flip": true,       # important: set flip to true
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
        "flip": true,               # it is correct, even if the mode is flip, we just count the forward metrics
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
        "sym_loss_wt": sym_loss_wt,

        "initializer": {
            "regexes": [
                [".*",
                    {
                        "type": "pretrained",
                        "weights_file_path": pretrained_path + "/best.th",
                        "parameter_name_overrides": {}
                    }
                ]
            ],
            "prevent_regexes": ["prevent_init_regex"]
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
            "num_serialized_models_to_keep": 2,
        },
        "validation_metric": "+fscore",
        "num_epochs": num_epochs,
        "cuda_device": cuda_device,
        "patience": 3,
    }
}