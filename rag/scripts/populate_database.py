import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

DATA_PATH = "rag/rag_data"
CHROMA_PATH = "rag/chroma"


def main():

    # check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()


    # create or update the data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list [Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=80,
        length=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # load the current database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function = get_embedding_function()
    )
    
    # calculate page ids
    chunks_with_ids = calculate_chunk_ids(chunks)

    # add or update the documents
    existing_items = db.get(include=[])  # ids are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    # creates ids like rag/rag_data/mentor_canada_resources/rag/rag_data
    # /mentor_canada_resources/1. SRDC. MENTOR.Final Report - Youth Results_FINAL - Copy.pdf:6:2"
    # page source : page number : chunk index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # if the page id is the same as the last one, increment index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
                current_chunk_index = 0
        
        # calculate the chunk id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # add to the page metadata
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()