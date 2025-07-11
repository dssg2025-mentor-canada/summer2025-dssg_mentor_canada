# make sure to pip install langchain-community

from langchain_ollama import OllamaEmbeddings

# for later (web deployment), will need to set up credentials and paid service
from langchain_community.embeddings.bedrock import BedrockEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name = "default", region_name = "us-east-1"
    # )
    return embeddings