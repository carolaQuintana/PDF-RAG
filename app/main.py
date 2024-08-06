from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .services import (
    generate_answer,
    debug_question,
    add_document,
    initialize_vector_store,
)

app = FastAPI()


class Query(BaseModel):
    question: str


class Request(BaseModel):
    page_title: str


doc_store = initialize_vector_store()


@app.get("/debug")
async def get_data_points(question: str):
    try:
        data_points = debug_question(question, doc_store)
        return {"Data points": data_points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ask")
async def ask_question(query: Query):
    try:
        answer = generate_answer(query.question, doc_store)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
async def embed_content(request: Request):
    try:
        result = add_document(request.page_title, doc_store)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
