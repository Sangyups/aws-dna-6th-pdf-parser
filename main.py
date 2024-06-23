import os
import shutil
from glob import glob

from langchain_community.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace

from utils.common_utils import load_pickle, to_pickle

image_path = "./fig"
file_path = "./data/traffic_law.pdf"


if os.path.isdir(image_path):
    shutil.rmtree(image_path)
os.mkdir(image_path)

loader = UnstructuredFileLoader(
    file_path=file_path,
    chunking_strategy="by_title",
    mode="elements",
    strategy="hi_res",
    hi_res_model_name="yolox",  # "detectron2_onnx", "yolox", "yolox_quantized"
    extract_images_in_pdf=True,
    # ['pdf', 'jpg', 'png', 'xls', 'xlsx', 'heic']
    # skip_infer_table_types='[]',
    pdf_infer_table_structure=True,  # enable to get table as html using tabletrasformer
    extract_image_block_output_dir=image_path,
    # False: to save image
    extract_image_block_to_payload=False,
    max_characters=4096,
    new_after_n_chars=4000,
    combine_text_under_n_chars=2000,
    languages=["kor+eng"],
    post_processors=[clean_bullets, clean_extra_whitespace],
)
docs = loader.load()
to_pickle(docs, "./data/parsed_unstructured.pkl")
docs = load_pickle("./data/parsed_unstructured.pkl")

tables, texts = [], []
images = glob(os.path.join(image_path, "*"))

for doc in docs:

    category = doc.metadata["category"]
    print(category)

    if category == "Table":
        tables.append(doc)
    elif category == "Image":
        images.append(doc)
    else:
        texts.append(doc)

    images = glob(os.path.join(image_path, "*"))

print(f" # texts: {len(texts)} \n # tables: {len(tables)} \n # images: {len(images)}")
