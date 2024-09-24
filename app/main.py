"""Router file"""
import struct
import io

import numpy as np
import redis
from flask import Flask, jsonify, request
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
from tika import parser


app = Flask(__name__)

# Redis connection
r = redis.Redis(host='redis', port=6379, db=0)

model = SentenceTransformer('all-MiniLM-L6-v2')

INDEX_NAME = "index4"
DOC_PREFIX = "doc:"
VECTOR_DIMENSIONS = 384

# Create Redis index for vector search
def create_index(vector_dimensions: int):
    """create_index func"""
    try:
        # check to see if index exists
        r.ft(INDEX_NAME).info()
        print("Index already exists!")
    except Exception as e:
        print("Exception in indexing reindexing",e)
        # schema
        schema = (
            TagField("tag"),                       # Tag Field Name
            VectorField("vector",                  # Vector Field Name
                "HNSW", {                          # Vector Index Type: FLAT or HNSW
                    "TYPE": "FLOAT32",             # FLOAT32 or FLOAT64
                    "DIM": vector_dimensions,      # Number of Vector Dimensions
                    "DISTANCE_METRIC": "COSINE",   # Vector Search Distance Metric
                }
            ),
        )

        # index Definition
        definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)

        # create Index
        r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)

# Convert vector to binary format
def vector_to_binary(vector):
    """Function for vector_to_binary"""
    return struct.pack(f'{len(vector)}f', *vector)

# Add a document with a vector to Redis
@app.route('/add', methods=['POST'])
def add_document():
    """Function for add doc."""
    data = request.json
    doc_id = data["id"]
    doc_tag = data['tag']
    content = data['content']
    embedding = model.encode(content)

    # Add document to Redis
    r.hset(f"doc:{doc_id}", mapping = {
        "vector": embedding.tobytes(),
        "content": content,
        "tag": doc_tag
    })
    return jsonify({"status": "Document added successfully"}), 201

# Search for the nearest neighbors
@app.route('/search', methods=['POST'])
def search_vector():
    """Function for search_vector."""
    data = request.json
    query_embedding = model.encode(data['query'])

    query = (
        Query("(@tag:{ ST })=>[KNN 3 @vector $vec as score]")
        .sort_by("score")
        .return_fields("tag", "score")
        # .paging(0, 2)
        .dialect(2)
    )
    query_params = {"vec": query_embedding.tobytes()}
    res = r.ft(INDEX_NAME).search(query, query_params).docs

    # Parse and return search results
    results = []
    for doc in res:
        results.append({
            "id": doc['id'],
            "score": doc['score']
        })

    return jsonify({"results": results})

def extract_text_with_tika(file):
    """Function to extract text """
    # extracted_text = ""
    # chunk_size = 1024 * 1024 # 1 KB chunks

    # while True:
    #     chunk = file.read(chunk_size)
    #     if not chunk:
    #         break
    #     print(f"chunk {chunk}",flush=True)
    #     response = parser.from_buffer(io.BytesIO(chunk))
    #     if response and response.get("content",""):
    #         extracted_text += response.get("content","")
    # return extracted_text
    parsed_file = parser.from_buffer(file)
    return parsed_file.get("content", "")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Function for upload."""

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    bucket_name = request.form.get('bucketName')
    obj_name = request.form.get('objName')
    obj_unique_path = bucket_name + "/" + obj_name
    print(f"obj_unique_path {obj_unique_path}",flush=True)

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    extracted_text = extract_text_with_tika(file)
    extracted_text = extracted_text.replace("\n", "")
    print(f"extracted text {extracted_text} ", flush=True)
    if extracted_text:

        embedding = model.encode(extracted_text)

        # Add document to Redis
        r.hset(f"doc:{obj_unique_path}", mapping = {
            "vector": embedding.tobytes(),
            "tag": "ST"
        })
        return jsonify({"message": "File successfully processed", "filename": file.filename}), 200

    return jsonify({"error": "Unable to extract text from the file"}), 400

# Start the Flask application
if __name__ == '__main__':
    create_index(vector_dimensions=VECTOR_DIMENSIONS)
    app.run(host='0.0.0.0', port=5007)
