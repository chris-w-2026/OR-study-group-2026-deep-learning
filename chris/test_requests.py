import requests
import trafilatura
from bs4 import BeautifulSoup

def clean_htmls(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return trafilatura.extract(str(soup), include_comments=False, include_tables=False)


if __name__ == "__main__":
    response = requests.get("https://en.wikipedia.org/wiki/Isolation_forest", headers={"user-agent": "Mozilla/5.0"})
    print(clean_htmls(response.text)[:1000])

