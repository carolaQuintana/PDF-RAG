from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_community.document_loaders import WikipediaLoader
from dotenv import load_dotenv, find_dotenv
import os
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings


load_dotenv(find_dotenv())


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings()


def initialize_vector_store():
    docs = WikipediaLoader(query="HUNTER X HUNTER", load_max_docs=1).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    doc_store = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        location=":memory:",
        collection_name="my_documents",
    )
    return doc_store


doc_store = initialize_vector_store()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def debug_question(question: str):
    results = doc_store.similarity_search(question, 2)
    data_points = [
        {"page_content": r.page_content, "metadata": r.metadata} for r in results
    ]
    return {"data_points": data_points}


def generate_answer(question: str) -> str:
    results = doc_store.similarity_search(question, 2)
    context = format_docs(results)

    content = f"""Answer the following question using the provided context. If you cannot answer the question within the context, don't lie and make up stuff. Just say you need more context.

Question: {question}

Context: {context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": content}]
    )

    return response.choices[0].message.content.strip()


def add_document(page_title: str) -> str:
    try:
        doc = WikipediaLoader(query=page_title, load_max_docs=1).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(doc)
        doc_store.add_documents(splits)
        return "Document added successfully"
    except Exception as e:
        raise Exception(f"Failed to add document: {str(e)}")
