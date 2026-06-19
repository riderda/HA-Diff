import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


from Diffusion.Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from Diffusion.Model_manba import ConditionalUNet1D
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])


    data = torch.load(modelConfig["data_path"])
    X = data['sequences']
    y = data['labels']

    L = X.shape[1]
    new_len = (L // 8) * 8
    if new_len < L:
        d = L - new_len
        left = d // 2
        right = d - left
        X = X[:, left: L - right, :]



    seq_len = X.shape[1]
    if modelConfig.get("seq_len") != seq_len:

        modelConfig["seq_len"] = seq_len


    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True
    )


    net_model = ConditionalUNet1D(
        T=modelConfig["T"],
        num_labels=modelConfig["num_labels"],
        seq_len=modelConfig["seq_len"],
        feature_dim=5,
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(
            os.path.join(modelConfig["save_dir"], modelConfig["training_load_weight"]),
            map_location=device
        ), strict=False)
        print("Model weight loaded.")

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4
    )

    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler
    )

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]
    ).to(device)


    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for sequences, labels in tqdmDataLoader:
                b = sequences.shape[0]
                optimizer.zero_grad()

                x_0 = sequences.to(device)
                labels = labels.to(device)


                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)

                loss = trainer(x_0, labels).sum() / (b ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"]
                )
                optimizer.step()

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "seq shape": f"{x_0.shape}",
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        warmUpScheduler.step()
        torch.save(net_model.state_dict(),
                   os.path.join(modelConfig["save_dir"], f'ckpt_{e}_.pt'))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])


    step = int(modelConfig["eval_size"] // modelConfig["num_labels"])
    labelList = []
    k = 1
    for i in range(1, modelConfig["eval_size"] + 1):
        labelList.append(torch.ones(size=[1]).long() * k)
        if i % step == 0 and k < modelConfig["num_labels"]:
            k += 1
    labels = torch.cat(labelList, dim=0).long().to(device)
    print("labels: ", labels)


    model = ConditionalUNet1D(
        T=modelConfig["T"],
        num_labels=modelConfig["num_labels"],
        seq_len=modelConfig["seq_len"],
        feature_dim=5,
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)
    ckpt = torch.load(os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print("Model loaded for evaluation.")


    sampler = GaussianDiffusionSampler(
        model, modelConfig["beta_1"], modelConfig["beta_T"],
        modelConfig["T"], w=modelConfig["w"]
    ).to(device)


    noisySeq = torch.randn(
        size=[modelConfig["eval_size"], modelConfig["seq_len"], 5],
        device=device
    )
    if modelConfig.get("save_noise", False):
        noise_np = noisySeq.cpu().numpy()
        np.save(os.path.join(modelConfig["sampled_dir"], modelConfig["sampledNoiseName"]), noise_np)


    with torch.no_grad():
        sampledSeqs = sampler(noisySeq, labels)

    sampled_np = sampledSeqs.cpu().numpy()


    if modelConfig.get("save_as_text", False):
        bases = ['A', 'T', 'C', 'G', '-']
        fasta_path = os.path.join(modelConfig["sampled_dir"], "generated_sequences.fasta")
        with open(fasta_path, 'w') as f:
            for i, seq in enumerate(sampled_np):

                label_name = "human" if labels[i].item() == 1 else "nonhuman"
                f.write(f">{label_name}_sample_{i}\n")

                seq_idx = np.argmax(seq, axis=-1)
                seq_str = ''.join([bases[idx] for idx in seq_idx])

                for j in range(0, len(seq_str), 80):
                    f.write(seq_str[j:j+80] + '\n')
        print(f"FASTA sequences saved to {fasta_path}")

    print("Sampling finished.")