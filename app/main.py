from flask import Flask, request, jsonify
import numpy as np
import struct

import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Redis connection
r = redis.Redis(host='redis', port=6379, db=0)

model = SentenceTransformer('all-MiniLM-L6-v2')

INDEX_NAME = "index4"
DOC_PREFIX = "doc:"   
VECTOR_DIMENSIONS = 384

# Create Redis index for vector search
def create_index(vector_dimensions: int):
    try:
        # check to see if index exists
        r.ft(INDEX_NAME).info()
        print("Index already exists!")
    except:
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
    return struct.pack(f'{len(vector)}f', *vector)

# Add a document with a vector to Redis
@app.route('/add', methods=['POST'])
def add_document():
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
    data = request.json
    query_embedding = model.encode(data['query'])
    doc_tag = data['tag']
    
    query = (
        Query("(@tag:{ ST })=>[KNN 3 @vector $vec as score]")
        .sort_by("score")
        .return_fields("content", "tag", "score")
        .paging(0, 2)
        .dialect(2)
    )
    query_params = {"vec": query_embedding.tobytes()}
    res = r.ft(INDEX_NAME).search(query, query_params).docs
    
    # Parse and return search results
    results = []
    for doc in res:
        results.append({
            "id": doc['id'],
            "content": doc['content'],
            "score": doc['score']
        })

    return jsonify({"results": results})

# Start the Flask application
if __name__ == '__main__':
    create_index(vector_dimensions=VECTOR_DIMENSIONS)
    app.run(host='0.0.0.0', port=5007)
