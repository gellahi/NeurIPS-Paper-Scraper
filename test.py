import aiohttp
import asyncio
import os
import csv
import logging
import ssl
import re
from bs4 import BeautifulSoup

BASE_URL = "https://papers.nips.cc"
DOWNLOAD_DIR = "downloaded_pdfs"
CSV_FILE = "papers.csv"

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a custom SSL context to disable certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def sanitize_filename(name):
    """Sanitize string to be used as a filename."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

async def fetch_html(session, url):
    """Fetch HTML content asynchronously with custom SSL context."""
    try:
        async with session.get(url, ssl=ssl_context) as response:
            return await response.text()
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

async def get_year_links(session):
    """Extract available year links from the main page."""
    url = BASE_URL + "/paper_files/paper/"
    html = await fetch_html(session, url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    return [BASE_URL + a['href'] for a in soup.select("a[href^='/paper_files/paper/']")]

async def get_paper_links(session, year_url):
    """Extract paper links for a given year."""
    html = await fetch_html(session, year_url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    return [BASE_URL + a['href'] for a in soup.select("ul.paper-list li a[href$='Abstract-Conference.html']")]

async def process_paper(session, paper_url):
    """Process a single paper page, extracting metadata and downloading the PDF."""
    html = await fetch_html(session, paper_url)
    if not html:
        return
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Attempt to extract the title from <h2>; if not found, fallback to the <title> tag
    title_tag = soup.find("h2")
    if title_tag:
        title = title_tag.text.strip()
    else:
        head_title = soup.find("title")
        if head_title:
            title = head_title.text.strip()
            # Remove common suffixes like " - Conference" if present
            title = re.sub(r'\s*[-|]\s*.*$', '', title)
        else:
            title = "Unknown Title"
    
    # Extract authors from <i> tags
    authors = ", ".join([i.text.strip() for i in soup.find_all("i")])
    
    # Extract PDF URL
    pdf_link_tag = soup.select_one("a[href$='Paper-Conference.pdf']")
    pdf_url = BASE_URL + pdf_link_tag['href'] if pdf_link_tag else None

    # Extract abstract: first try a <p> tag with class "abstract", then fallback to the first non-empty <p>
    abstract_tag = soup.find("p", class_="abstract")
    if abstract_tag:
        abstract = abstract_tag.text.strip()
    else:
        abstract = ""
        for p in soup.find_all("p"):
            text = p.text.strip()
            if text:
                abstract = text
                break

    # Extract year from paper_url using regex
    year_match = re.search(r'/paper_files/paper/(\d{4})/', paper_url)
    year = year_match.group(1) if year_match else ""
    
    # Set constant metadata values for fields not available on the page
    booktitle = "NeurIPS"
    editor = ""
    pages = ""
    publisher = ""
    volume = ""
    label = "NeurIPS"
    
    if pdf_url:
        await download_pdf(session, pdf_url, title)
    
    metadata = {
        "author": authors,
        "booktitle": booktitle,
        "editor": editor,
        "pages": pages,
        "publisher": publisher,
        "title": title,
        "url": paper_url,
        "volume": volume,
        "year": year,
        "abstract": abstract,
        "label": label,
        "pdf_url": pdf_url  # Included for reference; can be omitted if not needed
    }
    save_metadata(metadata)
    logging.info(f"Processed: {title}")

async def download_pdf(session, url, title):
    """Download and save a PDF file."""
    safe_title = sanitize_filename(title)
    filename = os.path.join(DOWNLOAD_DIR, f"{safe_title}.pdf")
    try:
        async with session.get(url, ssl=ssl_context) as response:
            with open(filename, "wb") as f:
                f.write(await response.read())
        logging.info(f"Downloaded: {title}")
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")

def save_metadata(metadata):
    """Append paper metadata to the CSV file."""
    file_exists = os.path.isfile(CSV_FILE)
    # Use a fixed list of fieldnames to ensure consistent header order
    fieldnames = ["author", "booktitle", "editor", "pages", "publisher", "title", "url", "volume", "year", "abstract", "label", "pdf_url"]
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metadata)

async def main():
    """Main function to orchestrate the scraping."""
    async with aiohttp.ClientSession() as session:
        year_links = await get_year_links(session)
        for year_url in year_links:
            paper_links = await get_paper_links(session, year_url)
            tasks = [process_paper(session, url) for url in paper_links]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
