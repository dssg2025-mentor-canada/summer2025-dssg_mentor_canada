import argparse
import os
from dotenv import load_dotenv

from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "rag/chroma"


PROMPT_TEMPLATE = """
You are a domain expert in mentorship. Answer the question based only on the following context:

{context}

--------------------------------

Answer the question based on the above context: {question}
"""
# get variables from .env file
load_dotenv()

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text:str):

    # load the database with the embeddings
    embedding_function = get_embedding_function()

    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    db = PineconeVectorStore(index=index, embedding=embedding_function, text_key="text")


    # search the database
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        print("No relevant documents retrieved for this query.")
    else:
        for doc, score in results:
            print(f"Score: {score}, ID: {doc.metadata.get('id')}, Content preview: {doc.page_content[:300]}")


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()

