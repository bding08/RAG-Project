from kfp import dsl
from kfp import compiler

@dsl.component(
    base_image='python:3.9-slim',
    packages_to_install=['langchain']
)
def chunk_text_with_langchain(
    input_texts: str, 
    chunk_size: int,
    chunk_overlap: int
) -> list:  
    from langchain.text_splitter import CharacterTextSplitter

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = splitter.split_text(input_texts)
    print(f"Split input text into {len(chunks)} chunks.")

    return chunks

@dsl.component(
    base_image='python:3.9-slim', 
    packages_to_install=[
        'sentence-transformers==2.2.2',
        'torch>=1.13.1',
        'numpy<2.0.0',
        'huggingface_hub==0.11.1',
        'tqdm' 
    ]
)
def generate_embeddings(
    input_texts: list,
    output_path: dsl.Output[dsl.Dataset],
    model_name: str = 'all-MiniLM-L6-v2',
):
    import json
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    if not input_texts:
        print("No input texts provided.")
        return

    # Generate embeddings, using cosine similarity, processes 32 texts at once however 
    # can be increased depending on compute availability.
    # We could specify the device to use (gpu, cpu, cuda...)
    embeddings = model.encode(input_texts, show_progress_bar=True, batch_size=32)

    # Prepare output data in NDJSON format
    # Sould look like: {"the_id": 0, "text": "test to be vectorized here", "embedding": []}
    with open(output_path.path, 'w') as f:
        for idx, (text, emb) in enumerate(zip(input_texts, embeddings)):
            document = {
                "the_id": idx,
                "text": text,
                "embedding": emb.tolist()
            }
            f.write(json.dumps(document) + '\n')
    

@dsl.component(packages_to_install=[
    'requests>=2.25.0'
    ]
)
def search(
    embeddings_path: dsl.Input[dsl.Dataset], 
    k: int, 
    index_name: str
) -> str:

    import requests
    import json 
    
    # fill in with opensearch base url
    OPENSEARCH_BASE_URL = "http://..."
    
    embeddings = []
    with open(embeddings_path.path, 'r') as f:
        for line in f:
            document = json.loads(line.strip())
            embeddings = document["embedding"]
    
    print("embeddings: " + str(embeddings))

    knn_query = {
        "query": {
            "knn": {
                "vector": {
                    "vector": embeddings,  # Vector to search
                    "k": k             # Number of neighbors
                }
            }
        }
    }
    
    # Make a POST request to the OpenSearch REST API
    url = f"{OPENSEARCH_BASE_URL}/{index_name}/_search"
    response = requests.post(url, json=knn_query)
    response_data = response.json()

    texts = []
    for hit in response_data.get('hits').get('hits'):
        
        texts.append(hit.get('_source').get('text'))

    return ", ".join(texts)

    # return response.json()

@dsl.component(
    base_image='python:3.9-slim',
    packages_to_install=[
        'torch>=1.11.0', 
        'transformers>=4.34.0', 
        'sentencepiece'
    ]
)
def generate_response(
    query_list: list, 
    background_info: str, 
    output_path: dsl.Output[dsl.Dataset],  # Added output parameter
    model_name: str,
    access_token: str
) -> str:
    """Generate a response using a transformer model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    query = " ".join(query_list)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, use_auth_token=access_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, use_auth_token=access_token
    )


    # Prepare input without padding
    input_text = f"Answer the query: {query} using this additional context: {background_info}\nResponse:"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  
        max_new_tokens=200,  
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_response = response.split("Response:")[-1].strip()

    # Write the response to a file
    with open(output_path.path, 'w') as f:
        f.write(final_response)

    return final_response


# Pipeline Component used to do test run. You will probably not use this but leaving just for insight.
@dsl.pipeline(
    name='e2e-pipeline-test',
    description='Testing end to end functionality'
)
def end_to_end_pipeline(
    input_texts: str,
    index_name: str,
    k_neighbors: int,
    access_token: str,
    embedding_model: str = 'all-MiniLM-L6-v2',
    generative_model: str = 'meta-llama/Llama-3.2-1B'
):
    chunk_task = chunk_text_with_langchain(
        input_texts=input_texts,
        chunk_size=20,
        chunk_overlap=10
    )

    generate_embeddings_task = generate_embeddings(
        input_texts=chunk_task.output,
        model_name=embedding_model
    )

    search_task = search(
        embeddings_path=generate_embeddings_task.output,
        k=k_neighbors,
        index_name=index_name
    )

    generate_response_task = generate_response(
        query_list=chunk_task.output,
        background_info=search_task.output,
        model_name=generative_model,
        access_token = access_token
    )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=end_to_end_pipeline,
        package_path='llama_pipeline.yaml'
    )