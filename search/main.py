"""Router file"""
import numpy as np
import redis
from flask import Flask, jsonify, request
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

# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008)
