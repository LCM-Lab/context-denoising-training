from typing import Literal, Optional
import torch
import torch.distributed as dist
import torch.optim as opt
import torch.utils.data as tud
import torch.optim.lr_scheduler as tol
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataclasses import dataclass
# from einops import rearrange,reduce,repeat
# from einops.layers.torch import Rearrange,Reduce
from transformers import PreTrainedTokenizer
from loomtrain.utils.distributed.torch import all_reduce
from torch.distributed.nn.functional import all_gather
from loomtrain.utils.wandb import WandbConfig
from loomtrain.utils.tensorboard import TensorboardConfig
from loomtrain.trainer.base import Trainer, TrainerConfig
from loomtrain.modeling.gpt import GPT, GPTCELoss
from loomtrain.strategy import DeepspeedStrategy
from loomtrain.dataset.sft import SFTDataset
from loomtrain.utils.distributed_sampler import (
    DistributedSampler, DistributedBucketSampler
)

from loomtrain.utils.sequence import mask2position
from loomtrain.utils.ring_attn import set_cu_seqlens_for_ring_attn

@dataclass
class CDTTrainerConfig(TrainerConfig):
    beta: float = 1e-6
    epsilon: float = None
    perturb_type: Literal["opposite", "other", "both"] = "opposite"


class CDTGPT(GPT):
    def forward_embedding(self,
                          inputs_embeds,
                          seq_lens,
                          attention_mask,
                          ring_attn_group,
                          gather = True):

        if gather:
            gathered_inputs_embeds = all_gather(inputs_embeds, 
                                                group = ring_attn_group)
            inputs_embeds = torch.concat(gathered_inputs_embeds, dim = 1)
            # .reshape(1, -1, last_dim)
            

        if ring_attn_group is None:
            position_ids = mask2position(attention_mask)
        else:
            inputs_embeds, attention_mask, position_ids = set_cu_seqlens_for_ring_attn(
                inputs_embeds, attention_mask, seq_lens, ring_attn_group
            )


        output = self.model(inputs_embeds = inputs_embeds, 
                            attention_mask = attention_mask, 
                            position_ids = position_ids)

        output["logits"] = output["logits"].to(torch.float32)

        return output


class CDTTrainer(Trainer):
    '''
    strategy will wrap the model and setup dataloader here
    '''
    def __init__(
      self,
      model: CDTGPT,
      train_dataset: SFTDataset,
      eval_dataset: SFTDataset,
      optimizer: opt.Optimizer,
      strategy: DeepspeedStrategy,
      config: TrainerConfig,
      tokenizer: PreTrainedTokenizer = None,
      save_hf_ckpt: bool = False,
      disable_ds_ckpt: bool = False,
      wandb_config: WandbConfig = None,
      tensorboard_config: TensorboardConfig = None     
    ):
        super().__init__(
            model = model,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            optimizer = optimizer,
            strategy = strategy,
            config = config,
            wandb_config = wandb_config,
            tensorboard_config = tensorboard_config
        )

        self.tokenzier = tokenizer
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt


        self.loss_fn = GPTCELoss(ring_attn_group = strategy.ring_attn_group)

    def fit(self, load_ckpt: bool = True):
        assert isinstance(self.train_dataloader.sampler, DistributedSampler) or \
            isinstance(self.train_dataloader.batch_sampler, DistributedBucketSampler)

        states = self.load_ckpt(load_ckpt = load_ckpt)
        step, update_steps_per_epoch, start_epoch, consumed_samples, total_tokens, loss_tokens = \
            self.strategy.restore_ckpt(
                states, self.train_dataloader, self.config
            )

        assert len(self.train_dataloader.dataset) and update_steps_per_epoch, \
            f"train_dataset_len: {len(self.train_dataloader.dataset)} < batch_size: {self.config.batch_size}"
        if self.config.eval_steps < 0:
            self.config.eval_steps = update_steps_per_epoch

        epoch_bar = tqdm(range(0, self.config.max_epochs), 
                         desc = "Train epoch",  initial = start_epoch,
                         disable = dist.get_rank() != 0)

        # scheduler_steps_per_epoch = len(self.train_dataloader.dataset) // len(self.train_dataloader.batch_size)

        loss = 0
        for epoch in range(start_epoch, self.config.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples = 0 if epoch > start_epoch else consumed_samples 
                )
            if isinstance(self.train_dataloader.batch_sampler, DistributedBucketSampler):
                self.train_dataloader.batch_sampler.set_epoch(
                    epoch, consumed_samples = 0 if epoch > start_epoch else consumed_samples 
                )

            inital_global_step = step // self.strategy.accumulated_gradient % update_steps_per_epoch

            step_bar = tqdm(range(update_steps_per_epoch),
                            desc = f"Train step of epoch {epoch}",
                            initial = inital_global_step,
                            disable = dist.get_rank() != 0)
            if self.config.enable_micro_bar:
                micro_bar = tqdm(range(self.strategy.accumulated_gradient),
                                desc = f"Micro Batch of Step {inital_global_step}", initial = 0,
                                disable = dist.get_rank() != 0)
            
            self.model.train()
            embedding_layer = self.model.model.get_input_embeddings()

            micro_cnt = 0
            for inputs, attention_masks, loss_masks, seq_lens in self.train_dataloader:
                inputs = inputs.to(torch.cuda.current_device())                
                attention_mask = attention_masks.to(torch.cuda.current_device())
                loss_mask = loss_masks.to(torch.cuda.current_device())
                
                if micro_cnt % self.strategy.accumulated_gradient == 0:
                    tmp_inputs, tmp_attention_mask, tmp_loss_mask, tmp_seq_lens = [], [], [], [] #for second FP/BP
                    tmp_embedding_grads = []
                    tmp_gpt_loss = []

                micro_cnt += 1
                tmp_inputs += [inputs]
                tmp_attention_mask += [attention_mask]
                tmp_loss_mask += [loss_mask]
                tmp_seq_lens += [seq_lens]

                inputs_embeds = embedding_layer(inputs)
                inputs_embeds = inputs_embeds.detach().requires_grad_(True)
                
                output = self.model.forward_embedding(
                    inputs_embeds, seq_lens, attention_mask, 
                    self.strategy.ring_attn_group, gather = False
                )

                _gpt_loss = self._get_loss_from_input_output(
                    inputs, output, attention_mask, loss_mask
                )

                tmp_gpt_loss += [_gpt_loss.item()]

                self.strategy.backward(_gpt_loss, self.model)

                tmp_embedding_grads += [inputs_embeds.grad]
                

                self.strategy.zero_grad(self.model)
                torch.cuda.empty_cache()

                step += 1

                if micro_cnt % self.strategy.accumulated_gradient: continue

                #second FP/BP
                loss = 0
                for (inputs, attention_mask, loss_mask, seq_lens, embedding_grad) in zip(
                    tmp_inputs, tmp_attention_mask, tmp_loss_mask, tmp_seq_lens, tmp_embedding_grads
                ):
                    inputs_embeds = embedding_layer(inputs)
                    
                    # inputs_embeds.retain_grad()
                    # inputs_embeds.requires_grad = True

                    with torch.no_grad():
                        perturbation = self._perturbation_from_embeds(
                            embedding_grad,
                            epsilon = self.config.epsilon,
                            perturb_type = self.config.perturb_type
                        )
                    
                    inputs_embeds += perturbation

                    adv_output = self.model.forward_embedding(
                        inputs_embeds, seq_lens, attention_mask, 
                        self.strategy.ring_attn_group, gather = False,
                    )

                    cdt_loss = self._get_loss_from_input_output(
                        inputs, adv_output, attention_mask, loss_mask
                    )

                    self.strategy.backward(cdt_loss, self.model)
                    self.strategy.optimizer_step(self.model)


                    loss += cdt_loss.item()
                    total_tokens += sum(seq_lens)
                    loss_tokens += loss_mask.int().sum().item()
                
                    if self.config.enable_micro_bar:
                        micro_bar.set_postfix(dict(cdt_loss = cdt_loss.item()))
                        micro_bar.update()

            
                logs_dict = dict(
                    cdt_loss = cdt_loss.item(),
                    lr = self.scheduler.get_last_lr()[0]
                )
                logs_dict = all_reduce(logs_dict)

                logs_dict["mean_loss"] = loss / self.strategy.accumulated_gradient
                
                step_bar.set_postfix(logs_dict)
                step_bar.update()
                
                global_step = step // self.strategy.accumulated_gradient

                if self.config.enable_micro_bar:
                    micro_bar = tqdm(range(self.strategy.accumulated_gradient),
                                    desc = f"Micro Batch of Step {global_step}", initial = 0,
                                    disable = dist.get_rank() != 0)

                tokens_dict = dict(
                    total_tokens = all_reduce(total_tokens) * self.strategy.ring_groups / 10**9,
                    loss_tokens = all_reduce(loss_tokens) * self.strategy.ring_groups / 10**9
                )

                visualized_dict = { **{f"train/{k}": v for k, v in \
                                    {**logs_dict, "global_step": global_step}.items()},
                                    **{f"train/{k}(B)": v for k, v in tokens_dict.items()}
                                }

                self.update_visualization(visualized_dict,
                                        global_step,
                                        self.config.logging_steps,
                                        step = global_step)
                finished = (epoch + 1 == self.config.max_epochs) \
                    and (global_step == update_steps_per_epoch)
                self.save_model(global_step = global_step, finished = finished)
                self.evaluate(global_step, finished = finished)
                self.save_ckpt(global_step = global_step, 
                            client_state = dict(
                                consumed_samples = global_step * self.config.batch_size,
                                ** tokens_dict
                                )
                            )
            
            epoch_bar.update()
    
        self.finish_visualization()

    def evaluate(self, global_step:int = 0, finished: bool = False):
        if global_step % self.config.eval_steps and (not finished): return
        self.model.eval()
        bar_dict = dict()
        with torch.no_grad():
            loss = 0
            total_tokens, loss_tokens = 0, 0
            step_bar = tqdm(
                range(len(self.eval_dataloader)),
                desc = f"Eval stage of steps {global_step}",
                disable = dist.get_rank() == 0,
            )
            for times, (inputs, attention_masks, loss_masks, seq_lens) in enumerate(self.eval_dataloader):
                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                loss_mask = loss_masks.to(torch.cuda.current_device())


                gpt_loss = self._get_loss(
                    inputs, attention_mask, loss_mask, seq_lens
                )

                loss += gpt_loss.item()
                total_tokens += sum(seq_lens)
                loss_tokens += loss_mask.int().sum().item()

                bar_dict = {"eval gpt_loss": loss / (times + 1)}

                step_bar.update()
            if not bar_dict: return
            logs = all_reduce(bar_dict)
            step_bar.set_postfix(logs)
            
            tokens_dict = dict(
                total_tokens = all_reduce(total_tokens, op = "sum") / 10**9,
                loss_tokens = all_reduce(loss_tokens, op = "sum") / 10**9
            )
            visualized_dict = { **{f"eval/{k}": v for k, v in {**logs, "global_step": global_step}.items()},
                                **{f"eval/{k}(B)": v for k, v in tokens_dict.items()}
                               }

            self.update_visualization(
                visualized_dict, 
                global_step, 
                self.config.logging_steps, 
                step = global_step
            )
        
        self.model.train()
    

    def _get_loss(self,
                  inputs: torch.LongTensor,
                  attention_mask: torch.LongTensor,
                  loss_mask: torch.BoolTensor,
                  seq_lens: list[int]):
        
        output = self.model(sequences = inputs, 
                            seq_lens = seq_lens,
                            attention_mask=attention_mask,
                            ring_attn_group = self.strategy.ring_attn_group)
        labels = torch.where(
            attention_mask.bool() & loss_mask.bool(),
            inputs,
            self.loss_fn.ignore_index
        )
        
        gpt_loss = self.loss_fn(output.logits, labels)

        return gpt_loss
    
    
    def _get_loss_from_input_output(self, 
                                    inputs: torch.LongTensor,
                                    output: torch.FloatTensor,
                                    attention_mask: torch.BoolTensor,
                                    loss_mask: torch.BoolTensor):
        labels = torch.where(
            attention_mask.bool() & loss_mask.bool(),
            inputs,
            self.loss_fn.ignore_index
        )
        
        gpt_loss = self.loss_fn(output.logits, labels)

        return gpt_loss   
    
    def _perturbation_from_embeds(self,
                                  inputs_embeds_grad,
                                  epsilon: float = None,
                                  perturb_type: Literal["opposite",
                                                        "other",
                                                        "both"] = "opposite"):
        
        dist.all_reduce(
            inputs_embeds_grad, 
            op = dist.ReduceOp.SUM,
            group = self.strategy.ring_attn_group
        )

        l2_grad = torch.norm(inputs_embeds_grad, p = 2, dim = -1)
        avg_grad = l2_grad.mean(dim = -1, keepdim = True)

        has_large_grad = l2_grad > avg_grad

        factor = torch.ones_like(l2_grad) * self.scheduler.get_last_lr()[0] \
            * self.config.beta if epsilon is None else epsilon / (1 + l2_grad)

        if perturb_type == "opposite":
            has_large_grad = ~has_large_grad
            factor = -factor

        if perturb_type == "both":
            perturbaiton = torch.where(
                has_large_grad.unsqueeze(-1),
                factor.unsqueeze(-1) * inputs_embeds_grad,  
                -factor.unsqueeze(-1) * inputs_embeds_grad,
            )
        else:
            perturbaiton = (factor * has_large_grad).unsqueeze(-1) * inputs_embeds_grad 
        
        return perturbaiton