from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
import uvicorn
from passage_retrieval_manager import Retriever, main  

app = FastAPI()

# Initialize parameters and retriever
args = argparse.Namespace(
    passages="mobile_eval_rag_retrieve/manager/passage.tsv",
    passages_embeddings="mobile_eval_rag_retrieve/manager/passage_00",
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

# Initialize retriever instance
retriever = Retriever(args)
retriever.setup_retriever()

class QueryRequest(BaseModel):
    query: str
    n_docs: int = 3  # Allow client to customize the number of results returned

@app.post("/retrieve")
async def retrieve_documents(request: QueryRequest):
    try:
        # Call the original main function logic
        result = retriever.search_document(request.query, request.n_docs)
        return {
            "results": [
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"]  
                } for doc in result
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)