from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer
from transformer_maskgit.videotextdataset import VideoTextDataset
from transformer_maskgit.train_transformer import TransformerTrainer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os
import yaml

with open("/home/jovyan/workspace/GenerateCT_saturn/paths.yaml", "r") as file:
    paths = yaml.safe_load(file)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def train():
    # set up distributed training

    ctvit = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=128,
        patch_size=16,
        temporal_patch_size=2,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )

    # Load the pre-trained weights

    pretrained_ctvit_path = paths["ctvit_checkpoint"]
    ctvit.load(pretrained_ctvit_path)

    maskgit = MaskGit(
        num_tokens=8192,
        max_seq_len=10000,
        dim=512,
        dim_context=768,
        depth=6,
    )

    transformer_model = MaskGITTransformer(ctvit=ctvit, maskgit=maskgit)

    transformer_model.load(paths["pretrained_models"] + "/transformer_pretrained.pt")

    # initialize DDP
    trainer = TransformerTrainer(
        transformer_model,
        data_folder=paths["all_inspect_data"],
        xlsx_file=paths["all_inspect_impressions"],
        num_train_steps=200000,
        batch_size=1,
        num_workers=1,
        pretrained_ctvit_path=paths["ctvit_checkpoint"],
        results_folder=paths["results_folder"] + "/transformer_train",
        save_results_every=100,
        save_model_every=500,
        accelerate_kwargs={
            "log_with": "wandb",
        },
    )

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # set up multiprocessing
    train()
