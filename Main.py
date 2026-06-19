from Diffusion.Train import train, eval

def main(model_config=None):

    modelConfig = {
        "state": "train",
        # "state": "eval",
        "epoch": 3000,
        "batch_size": 32,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 4],
        "num_res_blocks": 3,
        "dropout": 0.15,
        "lr": 1e-5,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "grad_clip": 1.0,
        "device": "cuda:1",


        "seq_len": 3000,
        "num_labels": 2,
        "feature_dim": 5,
        "data_path": "train_data.pt",


        "save_dir": "./CheckpointsDNA/",
        "training_load_weight": None,

        "eval_size": 2,
        "test_load_weight": "ckpt_1100_.pt",
        "sampled_dir": "./SampledSeqs/",
        "sampledNoiseName": "noise.npy",
        "sampledSeqName": "generated.npy",
        "w": 1.2,
        "save_noise": False,
        "save_as_text": True,
    }


    if model_config is not None:
        modelConfig.update(model_config)


    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()