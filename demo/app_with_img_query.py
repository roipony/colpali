import os
import sys

os.environ['HF_HOME'] = '/dccstor/ocr-ai/pony/projects/hf_home'
os.environ['HF_DATASETS_CACHE'] = '/dccstor/ocr-ai/pony/projects/hf_home/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/dccstor/ocr-ai/pony/projects/hf_home/models'
os.environ['HF_TOKEN'] = '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'

import gradio as gr
import torch
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

sys.path.insert(1, '/dccstor/ocr-ai/pony/projects/colpali/')

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries


def search(query: str, ds, images, uploaded_image):
    qs = []
    with torch.no_grad():
        # Check if an image was uploaded; if not, use the mock image
        if uploaded_image is not None:
            # Convert the uploaded image to a PIL image if necessary
            image_to_use = Image.open(uploaded_image) if isinstance(uploaded_image, str) else uploaded_image
        else:
            image_to_use = mock_image

        batch_query = process_queries(processor, [query], image_to_use)
        batch_query = {k: v.to(device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # Run evaluation
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(qs, ds)
    best_page = int(scores.argmax(axis=1).item())
    return f"The most relevant page is {best_page}", images[best_page]


def index(file, ds):
    """Example script to run inference with ColPali"""
    images = []
    for f in file:
        images.extend(convert_from_path(f))

    # Run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return f"Uploaded and converted {len(images)} pages", ds, images


COLORS = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]

# Load model
model_name = "vidore/colpali"
token = os.environ.get("HF_TOKEN")
model = ColPali.from_pretrained(
    "google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map="cuda", token=token
).eval()
model.load_adapter(model_name)
processor = AutoProcessor.from_pretrained(model_name, token=token)
device = model.device
mock_image = Image.new("RGB", (448, 448), (255, 255, 255))

with gr.Blocks() as demo:
    gr.Markdown("# ColPali: Efficient Document Retrieval with Vision Language Models 📚🔍")
    gr.Markdown("## 1️⃣ Upload PDFs")
    file = gr.File(file_types=["pdf"], file_count="multiple")

    gr.Markdown("## 2️⃣ Convert the PDFs and upload")
    convert_button = gr.Button("🔄 Convert and upload")
    message = gr.Textbox("Files not yet uploaded")
    embeds = gr.State(value=[])
    imgs = gr.State(value=[])

    # Define the actions
    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs])

    gr.Markdown("## 3️⃣ Search")
    query = gr.Textbox(placeholder="Enter your query here")
    uploaded_image = gr.Image(label="Upload an image (optional)")
    search_button = gr.Button("🔍 Search")
    message2 = gr.Textbox("Query not yet set")
    output_img = gr.Image()

    search_button.click(search, inputs=[query, embeds, imgs, uploaded_image], outputs=[message2, output_img])


if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True, server_name="0.0.0.0",
        server_port=7860,
        share=True)
