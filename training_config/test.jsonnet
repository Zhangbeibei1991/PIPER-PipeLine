local dataset = "tbd";
local lr = 0.001;
local num_epochs = 1;
local batch_size = 32;

{
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

        },
        "triplet": false,
        "flip": false,
        "collect_adj": true,
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

        },
        "triplet": false,
        "flip": true,
        "collect_adj": true,
    },

    "train_data_path": "data/" + dataset + "/train.json",

    "validation_data_path": "data/" + dataset + "/dev.json",

    "test_data_path": "data/" + dataset + "/test.json",

    "evaluate_on_test": true,

    "model": {
        "type": "logic_temporal",
        "classifier_type": "hdn",
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

            },
        },

        "encoder": {
            "type": "lstm",
            "input_size": 150,
            "hidden_size": 100,
            "num_layers": 1,
            "bidirectional": true,
        },

        "pair_encoder": {
            "type": "lstm",
            "input_size": 200,
            "hidden_size": 200,
            "num_layers": 1,
            "bidirectional": false,
        },

        "gcn_num_layers": 2,

        "source": dataset,

        "dropout_rate": 0.5,
    },

    "data_loader": {
        "batch_size": batch_size,
        "shuffle": true,
    },

    "trainer": {
        "type": "gradient_descent",
        "optimizer": {
            "type": "adam",
            "lr": lr,
        },
        "validation_metric": "+fscore",
        "num_epochs": num_epochs,
        "cuda_device": 0,
        "patience": 3
    }
}