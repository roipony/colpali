import os
import sys
import tempfile

# Set environment variables (replace with your own paths and tokens if necessary)
os.environ['HF_HOME'] = '/dccstor/ocr-ai/pony/projects/hf_home'
os.environ['HF_DATASETS_CACHE'] = '/dccstor/ocr-ai/pony/projects/hf_home/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/dccstor/ocr-ai/pony/projects/hf_home/models'
os.environ['HF_TOKEN'] = 'your_hf_token_here'  # Replace with your Hugging Face token
os.environ['GRADIO_TEMP_DIR'] = '/dccstor/ocr-ai/pony/projects/hf_home/tmp/gradio/'

from typing import List, cast

import gradio as gr
import torch
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add your project's directory to sys.path
sys.path.insert(1, '/dccstor/ocr-ai/pony/projects/colpali/')

from colpali_engine.models.paligemma import ColPali
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

from PyPDF2 import PdfReader, PdfWriter

import numpy as np
from io import BytesIO

# Import Grad-CAM and related utilities
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def overlay_gradcam_heatmap(model, processor, query, image):
    # Define the ViT configuration
    vit_config = {
        'resolution': 448,
        'patch_size': 16,
        'n_patch_per_dim': 28,
    }

    # Resize the image to square
    input_image_square = image.resize((vit_config['resolution'], vit_config['resolution']))

    # Preprocess the inputs
    input_text_processed = processor.process_text(query, add_special_tokens=True)
    input_image_processed = processor.process_image(input_image_square, add_special_prompt=True)

    # Move to device
    input_text_processed = {k: v.to(model.device) for k, v in vars(input_text_processed).items()}
    input_image_processed = {k: v.to(model.device) for k, v in vars(input_image_processed).items()}

    # Get the image tensor and make it require gradients
    image_tensor = input_image_processed['pixel_values']
    image_tensor.requires_grad = True

    # Prepare inputs for the model
    inputs = {'pixel_values': image_tensor}
    inputs.update(input_text_processed)

    # Define the target layer (adjust based on your model's architecture)
    target_layers = [model.vit.embeddings]  # For ViT models, you can try using the embeddings layer

    # Create a wrapper function for the model's forward pass
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            # Update the pixel_values in inputs
            inputs['pixel_values'] = x
            outputs = self.model(**inputs)
            # Compute similarity between text and image embeddings
            text_embeddings = outputs[0]  # Adjust index based on your model's output
            image_embeddings = outputs[1]  # Adjust index based on your model's output
            # Compute dot product as similarity score
            similarity = torch.sum(text_embeddings * image_embeddings, dim=-1)
            return similarity

    wrapped_model = ModelWrapper(model)

    # Initialize GradCAM
    cam = GradCAM(model=wrapped_model, target_layers=target_layers, use_cuda=(model.device.type == 'cuda'))

    # Compute Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)

    # Get the heatmap for the first image in the batch
    grayscale_cam = grayscale_cam[0]

    # Convert image to numpy array
    input_image_np = np.array(input_image_square).astype(np.float32) / 255.0

    # Generate heatmap on image
    visualization = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)

    # Convert visualization back to PIL Image
    blended_image = Image.fromarray(visualization)

    return blended_image

def search(query: str, ds, images, top_n: int):
    dataloader = DataLoader(
        dataset=ListDataset[str]([query]),
        batch_size=1,
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

    # Prepare images with Grad-CAM heatmap
    top_images_with_heatmap = []
    for idx in top_indices:
        img = images[idx]
        img_with_heatmap = overlay_gradcam_heatmap(model, processor, query, img)
        top_images_with_heatmap.append(img_with_heatmap)

    return (
        f"Top {top_n} relevant pages are {top_indices}",
        top_indices,
        top_images_without_heatmap,
        top_images_with_heatmap,
    )

def index(file, embeds):
    """Example script to run inference with ColPali"""
    images = []
    pdfWriter = PdfWriter()
    for f in file:
        images.extend(convert_from_path(f))
        reader = PdfReader(f)
        for page in reader.pages:
            pdfWriter.add_page(page)

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

def update_gallery(show_heatmap, images_without_heatmap, images_with_heatmap):
    if show_heatmap:
        return images_with_heatmap
    else:
        return images_without_heatmap

# Load model
base_model_name = "vidore/colpaligemma-3b-pt-448-base"
adapter_name = "vidore/colpali-v1.2"
device = get_torch_device("auto")

# Load model
model = ColPali.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()
model.load_adapter(adapter_name)
processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448"))

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
    show_heatmap_checkbox = gr.Checkbox(label="Show Grad-CAM heatmap", value=False)
    search_button = gr.Button("🔍 Search")
    message2 = gr.Textbox("Query not yet set")
    output_imgs = gr.Gallery(label="Top Relevant Images")
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

    # Define the generate function (you can implement your own logic here)
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

        # Save the temporary PDF to a file
        temp_file_path = "/tmp/temp_selected_pages.pdf"
        with open(temp_file_path, 'wb') as f:
            f.write(output_stream.read())

        # For demonstration, we'll just return a message and the path to the PDF
        response_text = f"Generated PDF for query: {query}"
        return response_text, temp_file_path

    # Define the actions for the generate button
    generate_button.click(
        generate,
        inputs=[query, pdf_writer_state, top_indices],
        outputs=[generate_message, output_pdf_view]
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True, server_name="0.0.0.0", server_port=7960, share=True)
