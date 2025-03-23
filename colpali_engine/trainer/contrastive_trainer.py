import torch
from transformers import Trainer
import os

from transformers import TrainerCallback
import torch
import numpy as np
import logging
import torch.distributed as dist


from datetime import datetime
class SaveLoRACheckpointCallback(TrainerCallback):
    """
    Callback to save LoRA adapters every K steps during training.
    """
    
    def __init__(self, save_steps=500, save_dir="lora_checkpoints", keep_recent_n=3, 
                 tokenizer=None, processor=None, config_file=None):
        """
        Initialize the callback.
        
        Args:
            save_steps (int): Save checkpoints every this many steps
            save_dir (str): Directory to save checkpoints
            keep_recent_n (int): Number of most recent checkpoints to keep (0 to keep all)
            tokenizer: HuggingFace tokenizer to save
            processor: Processor to save
            config_file (str): Path to config file to copy to checkpoint directory
        """
        self.save_steps = save_steps
        self.save_dir = save_dir
        self.keep_recent_n = keep_recent_n
        self.saved_checkpoints = []
        self.tokenizer = tokenizer
        self.processor = processor
        self.config_file = config_file
        
        # Get git hash at the beginning of training
        try:
            import subprocess
            self.current_git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception as e:
            print(f"Could not get git hash: {e}")
            self.current_git_hash = "git_hash_unavailable"
        
        # Create save directory if it doesn't exist
        if dist.get_rank() == 0 and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created checkpoint directory: {save_dir}")
            
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Event called at the end of a training step.
        """
        if args.local_process_index == 0 and state.global_step % self.save_steps == 0 and state.global_step > 0:
            # Create base directory if it doesn't exist
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            # Save the checkpoint using the custom save mechanism
            self._save_checkpoint(model, self.save_dir, step=state.global_step)
            
            # Keep track of which adapter dirs we've saved
            step_adapter_dir =  os.path.join(self.save_dir,"adapter_model", f"step_{state.global_step}")
            self.saved_checkpoints.append(step_adapter_dir)
            
            # Clean up old checkpoints if needed
            self._cleanup_old_checkpoints()
            
            print(f"Checkpoint process completed at step {state.global_step}")
            
    def _save_checkpoint(self, model, output_dir, step=None):
        """
        Custom save mechanism for the checkpoint.
        
        Args:
            model: The model to save
            output_dir: Base directory for saving
            step: Current training step number
        """
        # Create base directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save adapters with step number postfix
        if step is not None:
            adapter_dir =  os.path.join(output_dir,"adapter_model", f"step_{step}")
            if not os.path.exists(adapter_dir):
                os.makedirs(adapter_dir)
            model.save_pretrained(adapter_dir)

            print(f"Model adapters saved with step postfix to {adapter_dir}")
            
        model.save_pretrained(output_dir)
        print(f"Model adapters saved with to {output_dir}")
            
        # Save tokenizer to base directory if provided
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            print(f"Tokenizer saved to {output_dir}")
            
        # Save processor to base directory if provided
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
            print(f"Processor saved to {output_dir}")
            
        # Copy config file to base directory if provided
        if self.config_file is not None and os.path.exists(self.config_file):
            os.system(f"cp {self.config_file} {output_dir}/training_config.yml")
            print(f"Copied config file to {output_dir}/training_config.yml")
            
        # Save git hash to base directory
        with open(f"{output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
            print(f"Saved git hash to {output_dir}/git_hash.txt")

    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping only the most recent n.
        """
        if self.keep_recent_n > 0 and len(self.saved_checkpoints) > self.keep_recent_n:
            # Sort checkpoints by their step number (extracted from directory name)
            sorted_checkpoints = sorted(
                self.saved_checkpoints,
                key=lambda x: int(x.split("_step_")[-1]) if "_step_" in x else 0
            )
            
            # Get checkpoints to remove (keep the most recent n)
            checkpoints_to_remove = sorted_checkpoints[:-self.keep_recent_n]
            
            for checkpoint in checkpoints_to_remove:
                try:
                    import shutil
                    shutil.rmtree(checkpoint)
                    print(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    print(f"Failed to remove checkpoint {checkpoint}: {e}")
            
            # Update the list of saved checkpoints
            self.saved_checkpoints = sorted_checkpoints[-self.keep_recent_n:]

class GradientCheckCallback(TrainerCallback):
    def __init__(self, max_grad_norm=3.0, skip_bad_grads=True):
        self.max_grad_norm = max_grad_norm
        self.skip_bad_grads = skip_bad_grads
        self.grad_history = []
        self.skipped_steps = 0
        self.logger = logging.getLogger(__name__)
        
    def is_bad_gradient(self, grad_norm):
        """Check if gradient is problematic"""
        # Check for NaN or Inf
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            return True
            
        # Check for gradient explosion
        if grad_norm > self.max_grad_norm:
            return True
            
        # Check for abnormal spikes compared to recent history
        if len(self.grad_history) > 10:
            # Convert list to tensor on CPU for calculation
            recent_grads = torch.tensor(self.grad_history[-10:], device='cpu')
            recent_mean = torch.mean(recent_grads)
            if grad_norm > 5 * recent_mean:  # Gradient is 5x recent average
                return True
                
        return False
    
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Check gradients before the optimizer step"""
        model = kwargs.get("model", None)
        if model is None:
            return control
            
        # Calculate gradient norm
        grad_norm = self._get_grad_norm(model)
        # Only main process decides if gradient is bad
        skip_update = False
        # if args.local_process_index == 0:
        # Track gradient history - move to CPU and store as scalar
        grad_norm_item = grad_norm.detach().cpu().item()
        
        # Manage the history length
        if len(self.grad_history) < 1000:
            self.grad_history.append(grad_norm_item)
        else:
            self.grad_history.pop(0)
            self.grad_history.append(grad_norm_item)
    
        if self.is_bad_gradient(grad_norm):
            self.logger.warning(
                f"Bad gradient detected (norm={grad_norm:.4f}). " + 
                ("Skipping step." if self.skip_bad_grads else "Continuing anyway.")
            )
            
            if self.skip_bad_grads:
                self.skipped_steps += 1
                skip_update = True
        
        # # Broadcast skip decision to all processes if in distributed mode
        # if torch.distributed.is_initialized():
        #     skip_tensor = torch.tensor([1 if skip_update else 0], device=model.device)
        #     torch.distributed.broadcast(skip_tensor, src=0)
        #     skip_update = bool(skip_tensor.item())
        
        if skip_update:
            # Clear the bad gradients
            model.zero_grad()
            # Skip the optimizer step by setting control.should_optimizer_step = False            
            # Log metrics
            if args.local_process_index == 0:
                trainer = kwargs.get("trainer", None)
                if trainer and hasattr(trainer, "log"):
                    trainer.log({
                        "skipped_steps": self.skipped_steps,
                        "bad_grad_norm": grad_norm,
                    })
        
        return control
    
    def _get_grad_norm(self, model):
        """Calculate gradient norm for all parameters"""
        parameters = [p for p in model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return torch.tensor(0.0, device=model.device)
        
        norm_type = 2.0  # Using L2 norm
        norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), 
            norm_type
        )
        
            
        return norm

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add our custom metrics to the logs"""
        if logs is not None and args.local_process_index == 0:
            logs["skipped_steps"] = self.skipped_steps


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, output_dir, save_lora_every_n_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0
        self.output_dir = output_dir  # Output directory for saving LoRA adapters

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
        # feed only kwargs with 'doc_' prefix
        doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
        if "neg_doc_input_ids" in inputs:
            neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            # Check if loss is NaN, infinite, or exceeds threshold
            # if torch.isnan(loss) or torch.isinf(loss):
            #     loss = torch.tensor(float('nan')).to(loss.device)

            return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

        loss = self.loss_func(query_outputs, doc_outputs)
        # Check if loss is NaN, infinite, or exceeds threshold
        # if torch.isnan(loss) or torch.isinf(loss):
        #     loss = torch.tensor(float('nan')).to(loss.device)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
            if "neg_doc_input_ids" in inputs:
                neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None
