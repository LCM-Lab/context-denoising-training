import argparse, os
os.environ["CUDA_LAUNCH_BLOCKING"] ="1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

from datasets import load_from_disk
from loomtrain.utils.common.args import *
from loomtrain.strategy import (
    DeepspeedConfig, 
    RingAttnConfig, 
    DeepspeedStrategy
)
from loomtrain import cached, basename, path_join
from loomtrain.utils.wandb import WandbConfig
from loomtrain.utils.tensorboard import TensorboardConfig
from trainer import CDTTrainerConfig, CDTTrainer, CDTGPT
from loomtrain.modeling import GPT
from loomtrain.dataset.blended import BlendedSFTDataset
from loomtrain.dataset.sft import SFTDataset
from loomtrain.utils.init_hf import init_model, init_tokenizer
import torch.distributed as dist

def train(args):
    trainer_config = CDTTrainerConfig(
        batch_size = args.global_batch_size,
        bucket_size = 0,
        micro_batch_size = 1,
        max_epochs = args.max_epochs,
        scheduler = "cosine_with_min_lr",
        learing_rate = getattr(args, "learning_rate", 5e-6),
        lr_warmup_ratio = getattr(args, "lr_warmup_ratio", 0.03),
        save_steps = getattr(args, "ckpt_save_interval", -1),
        ckpt_dir = path_join(args.save_dir, args.save_name.rstrip("/") + "-ckpts"),
        max_ckpts = getattr(args,"max_ckpts", 1),
        weights_dir = path_join(args.save_dir, args.save_name.rstrip("/") + "-weights"),
        eval_steps = getattr(args, "evaluate_interval", 20),
        max_weights = args.max_weights,
        weights_saving_interval = args.weights_save_interval,
    )

    deepspeed_config = DeepspeedConfig(
        zero_stage = getattr(args, "zero_stage", 2),
        enable_bf16 = getattr(args, "enable_bf16", False),
        offload = getattr(args, "offload", False),
        adam_offload = getattr(args, "adam_offload", False),
        train_batch_size = trainer_config.batch_size,
        train_micro_batch_size_per_gpu = trainer_config.micro_batch_size,
        grad_clip = getattr(args, "grad_clip", 1.0),
        zpg = getattr(args,"zpg", 1),
        overlap_comm = getattr(args, "overlap_comm", False)
    )

    ring_attn_config = RingAttnConfig(
        ring_attn_size = getattr(args,"ring_attn_size", 8),
        ring_head_stride = getattr(args, "ring_head_stride", 4),    
    )

    if getattr(args, "wandb_api", None) is None:
        wandb_config = None
    else:
        wandb_config = WandbConfig(
            api_key = args.wandb_api,
            entity = args.wandb_entity,
            project = args.wandb_project,
            name = args.wandb_name,
            group = args.wandb_group,
            config = dict(**vars(trainer_config))
        )

    tensorboard_config = TensorboardConfig(
        log_dir = args.tensorboard_logdir,
        name = args.save_name
    )

    strategy = DeepspeedStrategy(
        seed = getattr(args, "seed", 42),
        deepspeed_config = deepspeed_config,
        ring_attn_config = ring_attn_config
    )

    strategy.init_distributed()

    if dist.get_rank() == 0:
        print("WORLD_SIE####     : ", dist.get_world_size())

    sft_model = init_model(getattr(args, "load_from", args.model_path))
    tokenizer =init_tokenizer(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = CDTGPT(sft_model)

    datasets_dicts = [
        load_from_disk(pth) for pth in args.dataset_paths
    ]

    if dist.get_rank() == 0:
        [
            cached(
                SFTDataset,
                kwargs = dict(
                    dataset = dataset["train"],
                    prompt_key = args.prompt_key, response_key = args.response_key,
                    tokenizer = tokenizer, max_length = args.max_data_length,
                    ring_attn_size = ring_attn_config.ring_attn_size
                ),
                cache_dir = args.data_cache_dir,
                cache_name = f"{basename(args.model_path)}={basename(data_pth)}=max-length{args.max_data_length}"
            ) for dataset,data_pth in zip(datasets_dicts, args.dataset_paths)
        ]

        [
            cached(
                SFTDataset,
                kwargs = dict(
                    dataset = dataset["eval"],
                    prompt_key = args.prompt_key, response_key = args.response_key,
                    tokenizer = tokenizer, max_length = args.max_data_length,
                    ring_attn_size = ring_attn_config.ring_attn_size
                ),
                cache_dir = args.data_cache_dir,
                cache_name = f"{basename(args.model_path)}={basename(data_pth)}=max-length{args.max_data_length}-eval"
            ) for dataset,data_pth in zip(datasets_dicts, args.dataset_paths)
        ]

    dist.barrier()

    sft_datasets =  [
            cached(
                SFTDataset,
                kwargs = dict(
                    dataset = dataset["train"],
                    prompt_key = args.prompt_key, response_key = args.response_key,
                    tokenizer = tokenizer, max_length = args.max_data_length,
                    ring_attn_size = ring_attn_config.ring_attn_size
                ),
                cache_dir = args.data_cache_dir,
                cache_name = f"{basename(args.model_path)}={basename(data_pth)}=max-length{args.max_data_length}"
            ) for dataset,data_pth in zip(datasets_dicts, args.dataset_paths)
        ]

    sft_datasets_eval =  [
            cached(
                SFTDataset,
                kwargs = dict(
                    dataset = dataset["eval"],
                    prompt_key = args.prompt_key, response_key = args.response_key,
                    tokenizer = tokenizer, max_length = args.max_data_length,
                    ring_attn_size = ring_attn_config.ring_attn_size
                ),
                cache_dir = args.data_cache_dir,
                cache_name = f"{basename(args.model_path)}={basename(data_pth)}=max-length{args.max_data_length}-eval"
            ) for dataset,data_pth in zip(datasets_dicts, args.dataset_paths)
        ]

    for d in sft_datasets:
        d.ring_attn_size = args.ring_attn_size

    train_dataset = BlendedSFTDataset(sft_datasets,
                                      sample_ratios = args.sample_ratios,
                                      sample_counts = args.sample_counts)
    eval_dataset = BlendedSFTDataset(sft_datasets_eval, 
                                     sample_counts = args.sample_counts_eval)

    trainer = CDTTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        optimizer =  strategy.setup_optimizer(model, 
                                              lr = args.learning_rate, 
                                              betas = args.adam_betas, 
                                              weight_decay = args.L2_weight_decay),
        strategy = strategy,
        config = trainer_config,
        tokenizer = tokenizer,
        wandb_config = wandb_config,
        tensorboard_config = tensorboard_config
    )

    trainer.fit(load_ckpt = args.do_resume)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank", type = int, default = -1
    )
    parser.add_argument(
        "--prompt-key", type = str, default = "chat_template"
    )

    parser.add_argument(
        "--response-key", type = str, default = "golden"
    )

    
    parser = add_saving_arguments(parser)
    parser = add_model_arguments(parser)
    parser = add_data_arguments(parser)
    parser = add_training_arguments(parser)

    args = parser.parse_args()

    train(args)