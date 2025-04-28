import os
from typing import List, Tuple, cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

USE_LOCAL_DATASET = os.environ.get("USE_LOCAL_DATASET", "1") == "1"


def add_metadata_column(dataset, column_name, value):
    def add_source(example):
        example[column_name] = value
        return example

    return dataset.map(add_source)


def load_train_set() -> DatasetDict:
    ds_path = "colpali_train_set"
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_dict = cast(DatasetDict, load_dataset(base_path + ds_path))
    return ds_dict


def load_train_set_detailed() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_train_set_with_tabfquad() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docmatix_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "docmatix-ir", split="train"))
    # dataset = dataset.select(range(100500))

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "Docmatix", "images", split="train"))

    return ds_dict, anchor_ds, "docmatix"

def load_wikiss() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "wiki-ss-nq", data_files="train.jsonl", split="train"))
    # dataset = dataset.select(range(400500))
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "wiki-ss-corpus", split="train"))

    return ds_dict, anchor_ds, "wikiss"


def load_train_set_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "manu/"
    dataset = cast(Dataset, load_dataset(base_path + "colpali-queries", split="train"))

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    print("Dataset size after filtering:", len(dataset))

    # keep only top 20 negative passages
    dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:20]})

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    anchor_ds = cast(Dataset, load_dataset(base_path + "colpali-corpus", split="train"))
    return ds_dict, anchor_ds, "vidore"


def load_train_set_with_docmatix() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
        "Docmatix_filtered_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot: List[Dataset] = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = concatenate_datasets(ds_tot)
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docvqa_dataset() -> DatasetDict:
    if USE_LOCAL_DATASET:
        dataset_doc = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="test"))
    else:
        dataset_doc = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test"))

    # concatenate the two datasets
    dataset = concatenate_datasets([dataset_doc, dataset_info])
    dataset_eval = concatenate_datasets([dataset_doc_eval, dataset_info_eval])
    # sample 100 from eval dataset
    dataset_eval = dataset_eval.shuffle(seed=42).select(range(200))

    # rename question as query
    dataset = dataset.rename_column("question", "query")
    dataset_eval = dataset_eval.rename_column("question", "query")

    # create new column image_filename that corresponds to ucsf_document_id if not None, else image_url
    dataset = dataset.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )
    dataset_eval = dataset_eval.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )

    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    return ds_dict


import os
from typing import Tuple
import logging

# datasets library is essential for Hugging Face datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def doc_fm_single_query_per_image(
    multi_query_ds_dict: DatasetDict,
    anchor_ds: Dataset,
    original_dataset_name: str
) -> Tuple[DatasetDict, Dataset, str]:
    """
    Derives the 'Step 1' dataset (one query per image) from a pre-loaded
    multi-query dataset by filtering for query_index == 0.

    Args:
        multi_query_ds_dict: The DatasetDict containing all queries (multiple per image).
                             Assumes 'train' split exists and has 'query_index' column.
        anchor_ds: The Dataset containing unique image anchors.
        original_dataset_name: The base name of the multi-query dataset.

    Returns:
        A tuple containing:
        - DatasetDict: The derived query dataset (one query per image).
        - Dataset: The original anchor dataset.
        - str: A new dataset name indicating it's the derived Step 1 version.
    """
    ds_corpus_name = f"{original_dataset_name}_filtered"
    logging.info(f"Deriving Step 1 dataset '{ds_corpus_name}'...")

    if 'train' not in multi_query_ds_dict or not multi_query_ds_dict['train']:
         logging.warning("Input multi_query_ds_dict['train'] is empty or missing. Returning empty Step 1 dataset.")
         empty_ds = Dataset.from_dict({})
         empty_dict = DatasetDict({'train': empty_ds, 'test': empty_ds})
         # Return original anchor_ds even if queries are empty
         return empty_dict, anchor_ds, f"{original_dataset_name}_step1_empty"

    all_queries_ds = multi_query_ds_dict['train']

    # Check if 'query_index' column exists
    if 'query_index' not in all_queries_ds.column_names:
        logging.error("Required 'query_index' column not found in multi_query_ds_dict['train']. Cannot derive Step 1 dataset.")
        # Return empty datasets to signal failure clearly
        empty_ds = Dataset.from_dict({})
        empty_dict = DatasetDict({'train': empty_ds, 'test': empty_ds})
        return empty_dict, Dataset.from_dict({}), f"{original_dataset_name}_step1_error" # Return empty anchor too

    # Filter to keep only the first query (index 0) for each image
    # This relies on the multi-query dataset having an entry with index 0 for every anchor.
    single_query_ds = all_queries_ds.filter(
        lambda example: example['query_index'] == 0,
        desc="Filtering for query_index 0" # Description for progress bar
    )

    num_filtered = len(single_query_ds)
    num_anchors = len(anchor_ds)

    if num_filtered == 0 and num_anchors > 0:
         logging.warning(f"Filtering for query_index=0 resulted in an empty dataset, but there are {num_anchors} anchors. Check data generation.")
    elif num_filtered != num_anchors:
        logging.warning(f"Mismatch after filtering: Queries ({num_filtered}) "
                        f"!= Anchors ({num_anchors}). Some anchors might lack a query with index 0.")
    total_samples = len(single_query_ds)

    # Calculate split index
    split_index = int(total_samples * 0.99)

    # First 99% for train
    train_dataset = single_query_ds.select(range(0, split_index))

    # Last 1% for test
    test_dataset = single_query_ds.select(range(split_index, total_samples))

    query_ds_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset # Reuse for test split as per previous pattern
    })

    anchor_ds = anchor_ds

    logging.info(f"Successfully derived dataset '{ds_corpus_name}' with {num_filtered} entries.")

    return query_ds_dict, anchor_ds, ds_corpus_name


def load_docfm():
    OUTPUT_DIR = "/dccstor/mm-rag/foad"
    DATASET_BASE_NAME = "docfm_multi_query_v1"
    CORPUS_QUERY_COLLATOR_BRANCH_NAME = "docfm"
    query_dataset_path = os.path.join(OUTPUT_DIR, f"{DATASET_BASE_NAME}_queries")
    anchor_dataset_path = os.path.join(OUTPUT_DIR, f"{DATASET_BASE_NAME}_anchors")
   
    multi_ds_dict = DatasetDict.load_from_disk(query_dataset_path)
    multi_anchor_ds = Dataset.load_from_disk(anchor_dataset_path)

    try:
        step1_ds_dict, step1_anchor_ds, step1_corpus_type = doc_fm_single_query_per_image(
            multi_query_ds_dict=multi_ds_dict,
            anchor_ds=multi_anchor_ds,
            original_dataset_name=CORPUS_QUERY_COLLATOR_BRANCH_NAME
        )
        print(f"\n📦 Derived Step 1 Dataset Name: {step1_corpus_type}")
        print(f"🔢 Step 1 Query dataset size ('train' split): {len(step1_ds_dict['train'])}")
        print(f"🖼️  Step 1 Anchor dataset size: {len(step1_anchor_ds)}") # Should match original anchor size
        return step1_ds_dict, step1_anchor_ds, step1_corpus_type
    except Exception as e:
        logging.error(f"Failed to derive Step 1 dataset: {e}", exc_info=True)
        return


class TestSetFactory:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __call__(self, *args, **kwargs):
        dataset = load_dataset(self.dataset_path, split="test")
        return dataset


if __name__ == "__main__":
    ds = TestSetFactory("vidore/tabfquad_test_subsampled")()
    print(ds)