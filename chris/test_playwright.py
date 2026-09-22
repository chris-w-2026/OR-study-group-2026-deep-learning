import asyncio
from test_requests import clean_htmls
from playwright.async_api import async_playwright

# "en.wikipedia.org/wiki/Isolation_forest"
urls = ["https://www.youtube.com/", "https://hmicfrs.justiceinspectorates.gov.uk/peel-assessments/peel-2018/?"]


async def fetch_html(browser, url):
    print(url)
    url = "https://" + url.removeprefix("https://")
    page = await browser.new_page()
    try:
        await page.goto(url, wait_until="networkidle")
        html = await page.content()
        return url, html
    finally:
        await page.close()


async def main(urls):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        tasks = [fetch_html(browser, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        await browser.close()

        return results


# Usage
results = asyncio.run(main(urls))

for result in results:
    if isinstance(result, Exception):
        print("Error:", result)
    else:
        url, html = result
        print(url, len(html))
        text = clean_htmls(html)
        if text is None:
            text = ""
        print(html)
        print("\n", "-" * 50, "\n")