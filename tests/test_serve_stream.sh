# generate, stream=false
# second example from ollama api doc

PORT=${PORT:-32988}

curl -s http://localhost:$PORT/api/generate -d '{
  "model": "openai/gpt-4",
  "stream": true,
  "prompt": "in 10 words or less, why is the sky blue?"
}'
