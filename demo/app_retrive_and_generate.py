import os
import sys
import tempfile

os.environ['HF_HOME'] = '/dccstor/ocr-ai/pony/projects/hf_home'
os.environ['HF_DATASETS_CACHE'] = '/dccstor/ocr-ai/pony/projects/hf_home/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/dccstor/ocr-ai/pony/projects/hf_home/models'
os.environ['HF_TOKEN'] = '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'
os.environ['GRADIO_TEMP_DIR']= '/dccstor/ocr-ai/pony/projects/hf_home/tmp/gradio/'
from typing import List, cast

import gradio as gr
import torch
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

import google.generativeai as genai

# temp_dir = tempfile.mkdtemp()  # This will create a temporary directory
# gr.routes.UPLOAD_FOLDER = temp_dir  

sys.path.insert(1, '/dccstor/ocr-ai/pony/projects/colpali/')

from colpali_engine.models.paligemma import ColPali
from colpali_engine.trainer.eval_utils import CustomRetrievalEvaluator as CustomEvaluator
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

from PyPDF2 import PdfReader, PdfWriter

import pprint
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import torch
from einops import rearrange
from PIL import Image
from tqdm import trange

import numpy as np

from io import BytesIO

def normalize_similarity_map_per_query_token(similarity_map: torch.Tensor) -> torch.Tensor:
    """Normalize the similarity map per query token."""
    # similarity_map shape: (1, n_text_tokens, h, w)
    similarity_map = similarity_map - similarity_map.amin(dim=(2, 3), keepdim=True)[0]
    similarity_map = similarity_map / (similarity_map.amax(dim=(2, 3), keepdim=True)[0] + 1e-8)
    return similarity_map

def overlay_similarity_heatmap(model, image, processor, output_text, output_image, query):
    # Define the ViT configuration
    vit_config = {
        'resolution': 448,
        'patch_size': 14,
        'n_patch_per_dim': 32,
    }

    # # Resize the image to square
    input_image_square = image.resize((vit_config['resolution'], vit_config['resolution']))
    
    tokens = processor.tokenizer.convert_ids_to_tokens(processor.process_queries([query])['input_ids'].numpy()[0])
    token_texts = [processor.tokenizer.convert_tokens_to_string([token]).strip() for token in tokens]
    # # Preprocess the inputs
    # input_text_processed = processor.process_queries([query])
    # input_image_processed = processor.process_images([input_image_square])
    
    # # Move to device
    # input_text_processed = {k: v.to(model.device) for k, v in input_text_processed.items()}
    # input_image_processed = {k: v.to(model.device) for k, v in input_image_processed.items()}
    
    # # Forward pass
    # with torch.no_grad():
    #     output_text = model(**input_text_processed)  # (1, n_text_tokens, hidden_dim)
    #     output_image = model(**input_image_processed)  # (1, n_patches + n_special_tokens, hidden_dim)
    
    # Remove special tokens from the output_image
    output_image = output_image.unsqueeze(0)
    output_text =torch.stack(output_text)
    output_image = output_image[:, : processor.image_seq_length, :]  # (1, n_patches, hidden_dim)
    
    # Reshape output_image to (1, h, w, c)
    output_image = rearrange(
        output_image, "b (h w) c -> b h w c", h=vit_config['n_patch_per_dim'], w=vit_config['n_patch_per_dim']
    )
    

    # Compute the dot product similarity map
    similarity_map = torch.einsum("bnk,bijk->bnij", output_text, output_image)  # shape: (b, n_text_tokens, h, w)

    # Compute the norms of the text and image embeddings
    # text_norms = torch.norm(output_text, dim=2)  # shape: (b, n_text_tokens)
    # image_norms = torch.norm(output_image, dim=3)  # shape: (b, h, w)

    # # Reshape norms for broadcasting
    # text_norms = text_norms[:, :, None, None]  # shape: (b, n_text_tokens, 1, 1)
    # image_norms = image_norms[:, None, :, :]   # shape: (b, 1, h, w)

    # # Compute the denominator for cosine similarity
    # denominator = text_norms * image_norms  # shape: (b, n_text_tokens, h, w)

    # # Avoid division by zero by adding a small epsilon
    # epsilon = 1e-8
    # denominator = denominator + epsilon

    # # Compute the cosine similarity map
    # similarity_map_normalized = similarity_map / denominator
    similarity_map_sum = similarity_map.mean(dim=1)[0].float() # (h, w)

    # Flatten the spatial dimensions (h, w) for argmax
    similarity_map_flat = similarity_map.view(similarity_map.shape[0], similarity_map.shape[1], -1)  # Shape: (b, n_text_tokens, h * w)

    # Compute the indices of the maximum values
    max_indices = similarity_map_flat.argmax(dim=2, keepdim=True)  # Shape: (b, n_text_tokens, 1)

    # Convert flat indices back to spatial indices
    max_h = max_indices // similarity_map.shape[3]  # Row index (height)
    max_w = max_indices % similarity_map.shape[3]  # Column index (width)

    # Create a binary mask for maximum values
    max_val_mask = torch.zeros_like(similarity_map, dtype=torch.bool)
    for b in range(similarity_map.shape[0]):  # Batch dimension
        for t in range(similarity_map.shape[1]):  # Text tokens
            max_val_mask[b, t, max_h[b, t].item(), max_w[b, t].item()] = True

    # # Get the similarity map
    # similarity_map = torch.einsum(
    #     "bnk,bijk->bnij", output_text, output_image
    # )  # (1, n_text_tokens, h, w)
    
    # # Normalize similarity map
    # similarity_map_normalized = normalize_similarity_map_per_query_token(similarity_map)  # (1, n_text_tokens, h, w)
    
    # Sum over the query tokens to get a single similarity map
    # similarity_map_sum = similarity_map_normalized.sum(dim=1)[0].float() # (h, w)
    
    # Upsample the similarity map to the image size
    similarity_map_upsampled = torch.nn.functional.interpolate(
        similarity_map_sum.unsqueeze(0).unsqueeze(0), size=input_image_square.size, mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    
    max_val_mask_upsampled = torch.nn.functional.interpolate(
        max_val_mask.float(), size=input_image_square.size, mode='nearest'
    ).squeeze().cpu().numpy()
    max_val_mask_upsampled = max_val_mask_upsampled.sum(axis=0)
    valid_mask = max_val_mask_upsampled > 0.  # Binary mask for valid values
    max_val_mask_upsampled = max_val_mask_upsampled / max_val_mask_upsampled.max()
    max_val_mask_upsampled[valid_mask] = max_val_mask_upsampled[valid_mask] * 0.5 + 0.5
    max_val_mask_upsampled[~valid_mask] = 0

# Normalize to [0, 1] by dividing by the number of tokens
    # max_val_mask_upsampled = max_val_mask_upsampled.max(axis=0)  # Aggregate across tokens to shape (h, w)
    # Normalize the similarity map to [0,1]
    # similarity_map_upsampled = (similarity_map_upsampled - similarity_map_upsampled.min()) / (similarity_map_upsampled.max() - similarity_map_upsampled.min() + 1e-8)

    # move the similarity map from [-1,1] to [0,1]
    similarity_map_upsampled +=1.
    similarity_map_upsampled = similarity_map_upsampled/2.
    
    # Convert to RGB heatmap
    import matplotlib.cm as cm
    heatmap = cm.jet(similarity_map_upsampled)  # (h, w, 4) RGBA
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB

    # Special color for maximum values
    max_val_color = np.array([0, 0, 255], dtype=np.uint8)  # Bright red color
    for i in range(3):  # Apply the color to the heatmap
        heatmap[..., i] = (max_val_mask_upsampled * max_val_color[i] + (1 - max_val_mask_upsampled) * heatmap[..., i]).astype(np.uint8)

    
    # Overlay the heatmap on the image
    heatmap_image = Image.fromarray(heatmap).convert("RGB").resize(image.size)
    blended_image = Image.blend(image.convert("RGB"), heatmap_image, alpha=0.4)
    return blended_image

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
    return file
  print("...all files ready")
  print()

def generate(query: str, pdf_writer, top_indices):
    # Create a temporary PDF from selected pages in pdf_writer
    output_stream = BytesIO()
    temp_writer = PdfWriter()

    # Adjust for 0-indexing and add selected pages to temp_writer
    for page_number in top_indices:
        temp_writer.add_page(pdf_writer.pages[page_number])

    # Write the temporary PDF to an in-memory byte stream
    temp_writer.write(output_stream)
    output_stream.seek(0)  # Reset the stream position to the start

    # Upload the temporary PDF to Gemini
    temp_file_path = "/tmp/temp_selected_pages.pdf"
    with open(temp_file_path, 'wb') as f:
        f.write(output_stream.read())

    files = [
        upload_to_gemini(temp_file_path, mime_type="application/pdf"),
    ]
    
    # Wait for the files to be ready
    wait_for_files_active(files)
    
    # Start the chat session
    chat_session = gen_model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    files[0],
                    query + "\n\n",
                ],
            }
        ]
    )
    
    response = chat_session.send_message(query)
    print(response.text)
    
    return response.text, temp_file_path

def search(query: str, ds, images, top_n: int, show_heatmap: bool):
    dataloader = DataLoader(
        dataset=ListDataset[str]([query]),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )
    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
        
    # Run scoring
    scores = processor.score(qs, ds).cpu().numpy()
    top_indices = scores.argsort(axis=1).flatten()[-top_n:][::-1].tolist()
    
    # Prepare images without heatmap
    top_images_without_heatmap = [images[i] for i in top_indices]

    # Prepare images with heatmap
    top_images_with_heatmap = []
    for idx in top_indices:
        img = images[idx]
        img_with_heatmap = overlay_similarity_heatmap(model, img, processor, qs, ds[idx], query)
        top_images_with_heatmap.append(img_with_heatmap)
    return f"Top {top_n} relevant pages are {top_indices}", top_indices, top_images_without_heatmap, top_images_with_heatmap


def index(file, ds):
    """Example script to run inference with ColPali"""
    images = []
    pdfWriter = PdfWriter()
    for f in file:
        images.extend(convert_from_path(f))
        reader = PdfReader(f)
        for page in reader.pages:
            pdfWriter.add_page(page)

    # run inference - docs
    # Run inference - docs
    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    
    ds: List[torch.Tensor] = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    return f"Uploaded and converted {len(images)} pages", ds, images, pdfWriter

GOOGLE_API_KEY = 'AIzaSyCtU-qGS6I7UoKngYN7cZ1CF1-fAWiZFVY'
genai.configure(api_key=GOOGLE_API_KEY)

# Create the model
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

gen_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

COLORS = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]
# Load model
# base_model_name = "vidore/colpaligemma-3b-pt-448-base"
# adapter_name = "vidore/colpali-v1.2"
device = get_torch_device("auto")

# Load model
from colpali_engine.models import ColQwen2, ColQwen2Processor

# model = ColQwen2.from_pretrained(
#         "vidore/colqwen2-v0.1",
#     torch_dtype=torch.bfloat16,
#     device_map=device,
#     ).eval()
# processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")


model_name = "vidore/colpali-v1.2"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,  # or "mps" if on Apple Silicon
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)



# model = ColPali.from_pretrained(
#     base_model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=device,
# ).eval()
# model.load_adapter(adapter_name)
# processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448"))
mock_image = Image.new("RGB", (448, 448), (255, 255, 255))
def toggle_textbox(checked):
    # Return a tuple; the first value is the visibility state of the textbox
    return gr.update(visible=checked)

with gr.Blocks() as demo:
    gr.Markdown("# ColPali: Efficient Document Retrieval with Vision Language Models 📚🔍")
    gr.Markdown("## 1️⃣ Upload PDFs")
    file = gr.File(file_types=["pdf"], file_count="multiple")

    gr.Markdown("## 2️⃣ Convert the PDFs and upload")
    convert_button = gr.Button("🔄 Convert and upload")
    message = gr.Textbox("Files not yet uploaded")
    embeds = gr.State(value=[])
    imgs = gr.State(value=[])
    pdf_writer_state = gr.State(value=None)

    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs, pdf_writer_state])

    gr.Markdown("## 3️⃣ Search")
    query = gr.Textbox(placeholder="Enter your query here")
    top_n_dropdown = gr.Dropdown(choices=[1, 3, 5, 10], label="Select number of top results", value=1)
    show_heatmap_checkbox = gr.Checkbox(label="Show similarity heatmap", value=False)
    search_button = gr.Button("🔍 Search")
    message2 = gr.Textbox("Query not yet set")
    output_imgs = gr.Gallery(label="Top Relevant Images",object_fit="contain")
    top_indices = gr.State(value=None)
    images_without_heatmap = gr.State()
    images_with_heatmap = gr.State()

    # Define the update_gallery function
    def update_gallery(show_heatmap, images_without_heatmap, images_with_heatmap):
        if show_heatmap:
            return images_with_heatmap
        else:
            return images_without_heatmap

    # When the search button is clicked, run the search and update the gallery
    search_button.click(
        search,
        inputs=[query, embeds, imgs, top_n_dropdown],
        outputs=[message2, top_indices, images_without_heatmap, images_with_heatmap]
    ).then(
        update_gallery,
        inputs=[show_heatmap_checkbox, images_without_heatmap, images_with_heatmap],
        outputs=output_imgs
    )

    # Update the gallery when the checkbox is toggled
    show_heatmap_checkbox.change(
        update_gallery,
        inputs=[show_heatmap_checkbox, images_without_heatmap, images_with_heatmap],
        outputs=output_imgs
    )

    gr.Markdown("## 4️⃣ Generate Response and View Selected PDF")
    generate_button = gr.Button("📝 Generate Response")
    generate_message = gr.Textbox(label="Generated Response")
    output_pdf_view = gr.File(label="View Generated PDF")

    # Define the actions for the generate button
    generate_button.click(
        generate, 
        inputs=[query, pdf_writer_state, top_indices], 
        outputs=[generate_message, output_pdf_view]
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True, server_name="0.0.0.0", server_port=7960, share=False)
