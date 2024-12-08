from kfp import dsl, compiler
from kfp.dsl import Input, Dataset, Output

#Component 1: Loading Documents and Chunking them
@dsl.component(packages_to_install=[
    "beautifulsoup4",
    "nest_asyncio",
    "tiktoken",
    "torch",
    "langchain",
    "langchain-community",
    "sentence-transformers"])
def scrape_and_chunk(
    urls: list,
    output_chunks: Output[Dataset],
):
    import nest_asyncio
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import SentenceTransformersTokenTextSplitter

    nest_asyncio.apply()

    # Asynchronous loading of multiple documents
    loader = WebBaseLoader(urls)
    loader.requests_per_second = 1
    docs = loader.aload()

    splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=50,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        tokens_per_chunk=128,
    )

    all_chunks = []
    for doc in docs:
        text_chunks = splitter.split_text(doc.page_content)
        all_chunks.extend(text_chunks)

    # Saving chunks to the kubeflow output path
    with open(output_chunks.path, "w") as f:
        for chunk in all_chunks:
            f.write(chunk + "\n")

# Component 2: Vectorize Chunks
@dsl.component(
    base_image='python:3.9-slim',
    packages_to_install=[
        'sentence-transformers==2.2.2',
        'torch>=1.13.1',
        'numpy<2.0.0',
        'huggingface-hub==0.11.1',
        'tqdm'
    ]
)
def generate_embeddings(
    input_chunks: Input[Dataset],
    output_embeddings: Output[Dataset],
    model_name: str = 'all-MiniLM-L6-v2',
):
    import json
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    # Reading chunks from input dataset
    with open(input_chunks.path, 'r') as f:
        chunks = f.readlines()

    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)

    # Write embeddings and documents in proper json format
    with open(output_embeddings.path, 'w') as f:
        for chunk, embedding in zip(chunks, embeddings):
            document = {
                "text": chunk.strip(),
                "vector": embedding.tolist()
            }
            f.write(json.dumps(document) + '\n')

# Component 3: Create OpenSearch Index
@dsl.component(packages_to_install=[
    'requests>=2.25.0',
    'kfp==2.10.1'
])
def create_index(
    index_name: str
) -> str:
    import requests
    #Add the Opensearch Base URL
    OPENSEARCH_BASE_URL = ""
    index_config = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 384,  # Adjust based on model's embedding size
                    "space_type": "l2",
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "parameters": {
                            "ef_construction": 100,
                            "m": 16
                        }
                    }
                },
                "text": {
                    "type": "keyword"
                }
            }
        }
    }

    url = f"{OPENSEARCH_BASE_URL}/{index_name}"
    response = requests.put(url, json=index_config)
    return f"Create Index Response: {response.status_code}, {response.json()}"

# Component 4: Bulk Add to OpenSearch
@dsl.component(packages_to_install=[
    'requests>=2.25.0',
    'kfp==2.10.1'
])
def bulk_index(
    input_data: Input[Dataset],
    index_name: str
) -> str:
    import requests
    import json
    #Add the Opensearch Base URL
    OPENSEARCH_BASE_URL = ""

    # Read the data to add from the input dataset
    with open(input_data.path, 'r') as f:
        data_to_add = [json.loads(line) for line in f.readlines()]

    # Prepare the bulk request data
    bulk_request = []
    for doc in data_to_add:
        bulk_request.append({"index": {"_index": index_name}})
        bulk_request.append(doc)

    # Convert the bulk request into NDJSON
    bulk_request_ndjson = "\n".join([json.dumps(item) for item in bulk_request]) + "\n"

    # Send the bulk request to OpenSearch
    response = requests.post(
        f"{OPENSEARCH_BASE_URL}/_bulk",
        data=bulk_request_ndjson,
        headers={"Content-Type": "application/x-ndjson"}
    )

    return f"Bulk Index Response: {response.status_code}, {response.json()}"

@dsl.pipeline(
    name='doc-ingestion-pipeline',
    description='Pipeline to upload documents and store them within our vector database'
)
def doc_ingestion_pipeline(
    urls: list,
    index_name: str
):
    # Step 1: Scrape and chunk documents
    scrape_and_chunk_task = scrape_and_chunk(urls=urls)

    # Step 2: Generate embeddings
    generate_embeddings_task = generate_embeddings(
        input_chunks=scrape_and_chunk_task.outputs["output_chunks"]
    )

    # Step 3: Create an index in OpenSearch
    create_index_task = create_index(index_name=index_name)

    # Step 4: Bulk add documents to OpenSearch
    bulk_index_task = bulk_index(
        input_data=generate_embeddings_task.outputs["output_embeddings"],
        index_name=index_name
    )
    bulk_index_task.after(create_index_task)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=doc_ingestion_pipeline,
        package_path="doc_ingestion_pipeline.yaml",
    )