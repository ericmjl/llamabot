# generate, stream=false
# second example from ollama api doc

PORT=${PORT:-32988}

curl -s http://localhost:$PORT/api/tags -d '{
}'
