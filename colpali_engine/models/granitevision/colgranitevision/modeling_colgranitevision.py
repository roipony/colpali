from typing import ClassVar, Optional

import torch
from torch import nn
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextConfig, LlavaNextPreTrainedModel

# from transformers.models.paligemma.modeling_paligemma import (
#     PaliGemmaConfig,
#     PaliGemmaForConditionalGeneration,
#     PaliGemmaPreTrainedModel,
# )


class ColGraniteVision(LlavaNextPreTrainedModel):
    """
    ColGraniteVision model implementation.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: LlavaNextConfig):
        super().__init__(config=config)

        model = LlavaNextForConditionalGeneration(config=config)
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model = model

        # TODO: Wait for ColPali2 to create a ColPaliConfig to allow specifying the embedding dimension.
        # We could do it now but it would break all the models trying to load the model from the checkpoint.
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)

        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

        outputs = self.model(*args, output_hidden_states=True, **kwargs)  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        
        attention_mask = kwargs["attention_mask"]
        if "pixel_values" in kwargs:
            input_ids = kwargs['input_ids']
            image_mask = (input_ids == self.config.image_token_index)
            # inputs_embeds = last_hidden_states.masked_scatter(image_mask)
            N, M = image_mask.shape
            # Create an index matrix: each row is 0, 1, ..., M-1
            idx = torch.arange(M, device=image_mask.device).expand(N, M)
            # Replace False positions with -1 so they are ignored by topk (since all valid indices are >=0)
            masked_idx = torch.where(image_mask, idx, torch.tensor(-1, device=image_mask.device))
            topk_values, _ = torch.topk(masked_idx, k=729, dim=1)
            last_k_indices, _ = torch.sort(topk_values, dim=1)
            last_k_indices_exp = last_k_indices.unsqueeze(-1).expand(-1, -1, last_hidden_states.size(-1))
            last_hidden_states = torch.gather(last_hidden_states, 1, last_k_indices_exp)
            attention_mask = torch.gather(attention_mask, 1, last_k_indices)
            
        attention_mask = attention_mask.unsqueeze(-1)

        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        # proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        proj = proj * attention_mask  # (batch_size, sequence_length, dim)

        return proj

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of=None,
    ) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings

        return model_embeds

    @property
    def patch_size(self) -> int:
        return self.model.vision_tower.config.patch_size
