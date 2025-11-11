from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import argparse, os, json
import uvicorn
import re
import traceback
import subprocess
import random
from PIL import Image
from passage_retrieval_operator import Retriever

app = FastAPI()

# Image directory settings
BASE_IMG_DIR = "mobile_eval_rag_retrieve/operator/app"
app.mount("/images", StaticFiles(directory=BASE_IMG_DIR), name="images")

# Default parameter initialization
args = argparse.Namespace(
    passages=None,
    passages_embeddings=None,
    n_docs=3,
    model_name_or_path="contriever-msmarco",
    per_gpu_batch_size=64,
    question_maxlength=512,
    no_fp16=False,
    lowercase=False,
    normalize_text=False,
    projection_size=768,
    n_subquantizers=0,
    n_bits=8,
    indexing_batch_size=1000000,
    save_or_load_index=False,
    output_dir=None,
    validation_workers=32,
    lang=None,
    dataset="none",
    local_rank=-1,
    rank=-1,
    world_size=-1,
    distributed_port=-1,
    data=None,
    temperature=1.0,
    train_data=None,
    eval_data=None,
    dedup=False,
    threads=1,
    print_answers=False,
    match="string",
    eval_single_thread=False,
    qa_pairs_format="pad",
    index=None,
    inner_batch_size=4096,
    retireve_only=False,
    use_gpu=True,
    save_index_path=None,
    load_index_path=None,
)

retriever_cache = {}  # Manage a separate retriever instance for each app

class QueryRequest(BaseModel):
    query: str
    n_docs: int = 3

def extract_app_name(query):
    match = re.search(r"App:/s*([/w/-]+)", query, re.IGNORECASE)
    return match.group(1) if match else None

def get_app_paths(app_name):
    app_dir = os.path.join(BASE_IMG_DIR, app_name)
    os.makedirs(app_dir, exist_ok=True)
    return {
        "app_dir": app_dir,
        "tsv": os.path.join(app_dir, "passage.tsv"),
        "img_dir": os.path.join(app_dir, "images"),
        "embedding_dir": os.path.join(app_dir, "embedding")
    }

def get_or_create_retriever(app_name, update_embedding=True):
    # If there is already a cache and no update is needed, directly return the cached retriever
    if app_name in retriever_cache and not update_embedding:
        return retriever_cache[app_name]

    paths = get_app_paths(app_name)
    tsv_path = paths["tsv"]
    embedding_dir = paths["embedding_dir"]
    embedding_file = os.path.join(embedding_dir, "passages_00")
    img_dir = paths["img_dir"]

    if not os.path.exists(img_dir):
        try:
            os.makedirs(img_dir, exist_ok=True)
            print(f"[INFO] Created 'images' directory for App={app_name}")
        except OSError as e:
            print(f"[ERROR] Failed to create 'images' directory: {e}")
            raise

    image_path = os.path.join(img_dir, "1.png")
    if not os.path.exists(image_path):
        try:
            # Generate a simple random image (1260x2800 pixels)
            width, height = 1260, 2800
            image = Image.new("RGB", (width, height))
            pixels = image.load()
            for i in range(width):
                for j in range(height):
                    # Set each pixel to a random RGB color
                    pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            image.save(image_path)
            print(f"[INFO] Generated and saved '1.png' for App={app_name}")
        except Exception as e:
            print(f"[ERROR] Failed to generate or save '1.png': {e}")
            raise
    
    # If passage.tsv does not exist, create it and write an initial entry
    if not os.path.exists(tsv_path):
        os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
        try:
            with open(tsv_path, "w", encoding="utf-8") as f:
                f.write("1/tImage: 1.png/ttest/n")  # Initial entry: 1 <tab> Image: 1.png <tab> test
            print(f"[INFO] Created new passage.tsv for App={app_name} and wrote initial entry.")
        except IOError as e:
            print(f"[ERROR] Failed to create passage.tsv: {e}")
            raise RuntimeError(f"Unable to create passage.tsv for App={app_name}")

    # Ensure the embedding directory exists
    os.makedirs(embedding_dir, exist_ok=True)

    # If the embedding file does not exist or needs to be updated, call the embedding generation script
    if not os.path.exists(embedding_file) or update_embedding:
        print(f"[INFO] {'Generating new embedding' if update_embedding else 'Embedding file not found'}, running generate_embedding.sh for App={app_name}")
        try:
            subprocess.run([
                "bash",
                "mobile_eval_rag_retrieve/operator/app/generate_embedding.sh",
                app_name
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding initialization failed for App={app_name}")

    # Set retriever parameters
    local_args = argparse.Namespace(**vars(args))
    local_args.passages = tsv_path
    local_args.passages_embeddings = embedding_file

    retriever = Retriever(local_args)
    retriever.setup_retriever()

    retriever_cache[app_name] = retriever
    return retriever

@app.post("/retrieve")
async def retrieve_documents(request: QueryRequest):
    try:
        app_name = extract_app_name(request.query)
        if not app_name:
            raise HTTPException(status_code=400, detail="Missing App field in Query")

        retriever = get_or_create_retriever(app_name)
        raw = retriever.search_document(request.query, request.n_docs)

        paths = get_app_paths(app_name)
        docs = []
        for doc in raw:
            full_path = doc["text"].split("Image:")[-1].strip()
            filename = os.path.basename(full_path)
            image_url = f"/images/{app_name}/images/{filename}"
            docs.append({
                "id": doc["id"],
                "query": doc["title"],
                "answer": doc["text"],
                "image_url": image_url
            })
        return {"results": docs}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/append_tsv")
async def append_tsv(
    instruction: str = Form(...),
    action_text: str = Form(...),
    screenshot: UploadFile = File(...)
):
    try:
        app_name = extract_app_name(instruction)
        if not app_name:
            raise HTTPException(status_code=400, detail="Missing App field in Instruction")

        paths = get_app_paths(app_name)
        tsv_path = paths["tsv"]
        embedding_dir = paths["embedding_dir"]
        embedding_file = os.path.join(embedding_dir, "passages_00")
        img_dir = paths["img_dir"]

        if not os.path.exists(tsv_path):
            last_id = 0
        else:
            with open(tsv_path, "r", encoding="utf-8") as f:
                lines = [l for l in f.readlines() if l.strip()]
                last_id = int(lines[-1].split("/t")[0]) if lines else 0

        new_id = last_id + 1
        image_filename = f"{new_id}.png"
        image_path = os.path.join(img_dir, image_filename)

        os.makedirs(img_dir, exist_ok=True)
                
        with open(image_path, "wb") as f_img:
            f_img.write(await screenshot.read())

        tsv_line = f"{new_id}/tAction: {action_text}. Image: {image_path}/tsubtask: {instruction}"
        with open(tsv_path, "a", encoding="utf-8") as f_tsv:
            f_tsv.write(tsv_line + "/n")

        if app_name in retriever_cache:
            retriever_cache[app_name].setup_retriever()

        retriever = get_or_create_retriever(app_name, update_embedding=True)

        return {"status": "success", "id": new_id}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)