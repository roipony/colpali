from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageOps
from transformers import BatchFeature, LlavaNextProcessor
import math

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


class ColGraniteVisionProcessor(BaseVisualRetrieverProcessor, LlavaNextProcessor):
    """
    Processor for ColPali.
    """

    visual_prompt_prefix: ClassVar[str] = "<|user|>\n<image>\nDescribe the image.\n"
    system_message: ClassVar[str] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    query_prefix: ClassVar[str] = "Query: "
    query_start: ClassVar[str] = "<|user|>\n"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = 14
        self.min_size = 384
        self.max_size = 384*2


    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.
        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    @staticmethod
    def smart_resize_helper(
        width: int,
        height: int,
        factor: int,
        min_size: int,
        max_size: int
    ) -> Tuple[int, int]:
        """
        Returns the resized image dimensions such that:
        1. The smaller dimension is set to 'min_size'.
        2. The larger dimension is scaled proportionally to maintain aspect ratio.
        3. If the larger dimension exceeds 'max_size', it is clipped to 'max_size',
        and the smaller dimension is adjusted accordingly to maintain aspect ratio.
        4. Both dimensions are divisible by 'factor'.
        """

        # Determine scale factor based on min_size
        if height < width:
            scale_factor = min_size / height
        else:
            scale_factor = min_size / width

        new_width = round(width * scale_factor)
        new_height = round(height * scale_factor)

        # If the longer dimension exceeds max_size, adjust accordingly
        if max(new_width, new_height) > max_size:
            clip_factor = max_size / max(new_width, new_height)
            new_width = round(new_width * clip_factor)
            new_height = round(new_height * clip_factor)

        # Ensure dimensions are divisible by factor
        # new_width = round_by_factor(new_width, factor)
        # new_height = round_by_factor(new_height, factor)

        return new_width, new_height
    
    @staticmethod
    def pad_image_center(image: Image.Image,
                        target_width: int,
                        target_height: int,
                        fill_color=(0, 0, 0)) -> Image.Image:
        """
        Pads the given image to be centered within the target dimensions.
        
        :param image: PIL Image to be padded.
        :param target_width: The desired width after padding.
        :param target_height: The desired height after padding.
        :param fill_color: Background color (default is black).
        :return: Padded image with centered content.
        """

        # Get original image size
        img_width, img_height = image.size

        # Compute padding values
        pad_left = (target_width - img_width) // 2
        pad_top = (target_height - img_height) // 2
        pad_right = target_width - img_width - pad_left
        pad_bottom = target_height - img_height - pad_top

        # Apply padding
        padded_image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill_color).convert("RGB")
        
        return padded_image


    def smart_resize(self, image: Image.Image) -> Image.Image:
        """
        Resize and convert the image to the required format.
        """
        image_size = image.size
        resized_height, resized_width = self.smart_resize_helper(
            width=image_size[0],
            height=image_size[1],
            factor=self.factor,
            min_size=self.min_size,
            max_size=self.max_size
        )
        return image.convert("RGB").resize((resized_width, resized_height))

    def smart_resize_and_pad(self, image: Image.Image) -> Image.Image:
        """
        Resize and pad the image to the required format.
        """
        return self.resize_and_pad_centered(
            image=image,
            factor=self.factor,
            min_size=self.min_size,
            max_size=self.max_size,
            fill_color=0
        )

    def resize_and_pad_centered(self,
        image: Image.Image, 
        factor: int, 
        min_size: int, 
        max_size: int, 
        fill_color=0
    ) -> Image.Image:
        """
        Resizes and pads an image such that:
        - The short side is set to `min_size`.
        - The long side is scaled proportionally but clipped to `max_size`.
        - The image is centered within the final padded area.
        
        :param image: PIL Image
        :param factor: Factor to make dimensions divisible by
        :param min_size: Minimum size for the short side
        :param max_size: Maximum allowed size for the long side
        :param fill_color: Background padding color (default black)
        :return: Resized and padded image
        """

        # Get original size
        width, height = image.size

        # Determine scale factor based on the short side (min_size)
        if width < height:
            scale_factor = min_size / width
            target_width = min_size
            max_scale_factor = min(max_size / height, scale_factor)
            target_height = round(height * max_scale_factor)
        else:
            scale_factor = min_size / height
            target_height = min_size
            max_scale_factor = min(max_size / width, scale_factor)
            target_width = round(width * max_scale_factor)

        # Ensure the longer side does not exceed max_size
        # if max(target_width, target_height) > max_size:
        #     clip_factor = max_size / max(target_width, target_height)
        #     target_width = round(target_width * clip_factor)
        #     target_height = round(target_height * clip_factor)

        # Ensure dimensions are divisible by factor
        # target_width = round_by_factor(target_width, factor)
        # target_height = round_by_factor(target_height, factor)

        # Resize the image
        resized_image = image.resize((target_width, target_height), Image.LANCZOS)

        # Determine final padded dimensions (aligned to short side)
        if width < height:
            final_width, final_height = min_size, max_size
        else:
            final_width, final_height = max_size, min_size

        # Compute padding to center the image
        pad_left = (final_width - target_width) // 2
        pad_top = (final_height - target_height) // 2
        pad_right = final_width - target_width - pad_left
        pad_bottom = final_height - target_height - pad_top

        # Apply centered padding
        # final_image = ImageOps.expand(resized_image, (pad_left, pad_top, pad_right, pad_bottom), fill_color).convert("RGB")
        final_image =resized_image.convert("RGB")

        return final_image

    def format_data(self, question, image ):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]
    
    def format_data_wo_role(self, question, image=None ):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColPali.
        """
        # texts_doc = [self.apply_chat_template(self.format_data_wo_role(self.visual_prompt_prefix, img),tokenize=False ) for img in images]
        texts_doc = [self.visual_prompt_prefix for _ in images]
        images = [self.smart_resize_and_pad(image) for image in images]

        batch_doc = self(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColPali.
        """
        # texts_query = [self.apply_chat_template(self.format_data(querie, None),tokenize=False ) for querie in queries]

        # batch_query = self(
        #     text=texts_query,
        #     images=[],
        #     return_tensors="pt",
        #     padding="longest",
        # )
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts_query: List[str] = []

        for query in queries:
            query = self.query_start + self.query_prefix + query
            query += suffix  # add suffix (pad tokens)

            # NOTE: Make input ISO to PaliGemma's processor
            query += "\n"

            texts_query.append(query)
        
        batch_query = self(
            text=texts_query,
            images= None,
            return_tensors="pt",
            padding="longest",
        )


        # batch_query = self.tokenizer(
        #     texts_query,
        #     text_pair=None,
        #     return_token_type_ids=False,
        #     return_tensors="pt",
        #     padding="longest",
        #     max_length=max_length,
        # )

        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
    ) -> Tuple[int, int]:
        n_patches_x = self.image_processor.size["width"] // patch_size
        n_patches_y = self.image_processor.size["height"] // patch_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
