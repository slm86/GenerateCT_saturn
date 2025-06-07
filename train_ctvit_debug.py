import torch
from transformer_maskgit import CTViT, CTViTTrainer
import yaml
import torch.distributed as dist

with open("/home/jovyan/workspace/GenerateCT_saturn/paths.yaml", "r") as file:
    paths = yaml.safe_load(file)

cvivit = CTViT(
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

# Resume training from checkpoint
cvivit.load(paths["pretrained_models"] + "/ctvit_pretrained.pt")

trainer = CTViTTrainer(
    cvivit,
    folder=paths["debug_data"],
    batch_size=8,
    num_workers=8,
    results_folder=paths["results_folder"] + "/ctvit",
    grad_accum_every=1,
    train_on_images=False,  # you can train on images first, before fine tuning on video, for sample efficiency
    use_ema=False,  # recommended to be turned on (keeps exponential moving averaged cvivit) unless if you don't have enough resources
    num_train_steps=10,
    num_frames=2,
    accelerate_kwargs={
        "log_with": "wandb",
    },
)

trainer.train()  # reconstructions and checkpoints will be saved periodically to ./results

if dist.is_initialized():
    dist.destroy_process_group()
