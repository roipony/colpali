from random import randint
from typing import Any, Dict, List, Optional

from PIL import Image

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class CorpusQueryCollator(VisualRetrieverCollator):
    """
    Collator for BEIR-style dataset training or hard negative training.

    This collator supports multiple corpus formats. The supported formats are:
    - "wikiss" (queries and qrels: Tevatron/wiki-ss-nq, corpus: Tevatron/wiki-ss-corpus)
    - "docmatix" (HuggingFaceM4/Docmatix)
    - "vidore" (vidore/colpali_train_set)
    - "docfm_filtered"  (DocFM corpus with doc_key and image_path - filtered such that we are using query[0] for each image)
    """

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        image_dataset: Optional["Dataset"] = None,  # noqa: F821
        mined_negatives: bool = True,
        corpus_format: str = "wikiss",
    ):
        super().__init__(processor=processor, max_length=max_length)
        if image_dataset is None:
            raise ValueError("`image_dataset` must be provided")
        self.image_dataset = image_dataset
        self.mined_negatives = mined_negatives
        self.corpus_format = corpus_format

        # Build docid_to_idx map for each supported corpus
        if self.corpus_format == "wikiss":
            print("Mapping docids to indices for wikiss")
            self.docid_to_idx = {docid: idx for idx, docid in enumerate(self.image_dataset["docid"])}
        elif self.corpus_format == "docmatix":
            # index by integer position for docmatix
            self.docid_to_idx = None  # direct indexing will be used
        elif self.corpus_format == "vidore":
            self.docid_to_idx = None  # integer string index
        elif self.corpus_format == "docfm_filtered":
            print("Mapping doc_keys to indices for docfm_filtered")
            self.docid_to_idx = {row['doc_key']: idx for idx, row in enumerate(self.image_dataset)}
        else:
            raise NotImplementedError(f"Corpus format {self.corpus_format} not supported in init")

    def get_image_from_docid(self, docid: str) -> Image.Image:
        """
        Returns the image corresponding to the given docid.
        """
        if self.corpus_format == "wikiss":
            return self.image_dataset[self.docid_to_idx[docid]]["image"]
        elif self.corpus_format == "docmatix":
            doc_id_str, doc_page_str = docid.split("_", 1)
            doc_id = int(doc_id_str)
            doc_page = int(doc_page_str)
            return self.image_dataset[doc_id]["images"][doc_page]
        elif self.corpus_format == "vidore":
            return self.image_dataset[int(docid)]["image"]
        elif self.corpus_format == "docfm_filtered":
            # load image from path
            row = self.image_dataset[self.docid_to_idx[docid]]
            return Image.open(row["image_path"]).convert("RGB")
        else:
            raise NotImplementedError(f"Corpus format {self.corpus_format} not supported in get_image")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_examples: List[Dict[str, Any]] = []

        for example in examples:
            # Extract positive document id depending on the corpus format.
            if self.corpus_format in {"wikiss", "docmatix"}:
                pos_docid = example["positive_passages"][0]["docid"]
            elif self.corpus_format == "vidore":
                pos_docid = example["positive_passages"][0]
            elif self.corpus_format == "docfm_filtered":
                pos_docid = example["doc_key"]
            else:
                raise NotImplementedError(f"Corpus format {self.corpus_format} not supported in call")

            sample = {
                "image": self.get_image_from_docid(pos_docid),
                "query": example.get("query", example.get("question", None)),
            }

            # Handle hard negatives if applicable
            if self.mined_negatives and self.corpus_format not in {"docfm_filtered"}:
                negative_candidates = example.get("negative_passages", [])
                if not negative_candidates:
                    raise ValueError("No negative passages available")
                chosen_negative = negative_candidates[randint(0, len(negative_candidates) - 1)]
                if self.corpus_format in {"wikiss", "docmatix"}:
                    negative_docid = chosen_negative["docid"]
                elif self.corpus_format == "vidore":
                    negative_docid = chosen_negative
                else:
                    raise NotImplementedError(f"Corpus format {self.corpus_format} not supported for negatives")
                sample["neg_image"] = self.get_image_from_docid(negative_docid)

            processed_examples.append(sample)

        return super().__call__(processed_examples)