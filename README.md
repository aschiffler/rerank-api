Rerank Endpoint (using curl):

Bash

curl -X POST "http://localhost:11435/rerank" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What is the capital of France?",
           "documents": [
             "Paris is the capital of France.",
             "The Eiffel Tower is in Paris.",
             "Berlin is the capital of Germany."
           ]
         }'
You should get a JSON response with scores, e.g.:

JSON

{"scores": [0.99876, 0.54321, 0.12345]}