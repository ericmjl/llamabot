"""Automatically update the list of Ollama models in llamabot/bot/ollama_model_names.txt"""

# /// script
# dependencies = [
#     "beautifulsoup4",
#     "lxml",
#     "requests"
# ]
# ///

from bs4 import BeautifulSoup
import requests

# Your HTML snippet
response = requests.get("https://ollama.ai/library?sort=newest")
if response.status_code == 200:
    html_content = response.text

    # Parse the HTML snippet with BeautifulSoup
    soup = BeautifulSoup(html_content, "lxml")

    # Find all h2 tags that contain the model names
    model_names = []
    for h2 in soup.find_all("h2"):  # Assuming h2 tags contain model names:
        try:
            model_name = h2.span.text.strip("\n").strip(" ").strip("\n")
            model_names.append(model_name)
        except AttributeError:
            pass  # Skip if h2 tag does not contain a span tag


# Write model names to llamabot/bot/ollama_model_names.txt
with open("llamabot/bot/ollama_model_names.txt", "w") as f:
    f.write("\n".join(model_names))
    f.write("\n")
