"""Automatically update the list of Ollama models in llamabot/bot/ollama_model_names.txt"""
from bs4 import BeautifulSoup
import requests

# Your HTML snippet
response = requests.get("https://ollama.ai/library?sort=newest")
if response.status_code == 200:
    html_content = response.text

    # Parse the HTML snippet with BeautifulSoup
    soup = BeautifulSoup(html_content, "lxml")

    # Find all h2 tags that contain the model names
    model_names = [
        h2.text.strip("\n").strip(" ").strip("\n") for h2 in soup.find_all("h2")
    ]

# Write model names to llamabot/bot/ollama_model_names.txt
with open("llamabot/bot/ollama_model_names.txt", "w") as f:
    f.write("\n".join(model_names))
    f.write("\n")
