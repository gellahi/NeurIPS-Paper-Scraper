import aiohttp
import asyncio
import os
import csv
import logging
import ssl
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from asyncio import Semaphore

BASE_URL = "https://papers.nips.cc"
DOWNLOAD_DIR = "downloaded_pdfs"
CSV_FILE = "papers.csv"
MAX_CONCURRENT_REQUESTS = 3  # Reduced from 5 to avoid overwhelming the server
TIMEOUT = aiohttp.ClientTimeout(total=30)  # 30 second timeout

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; NeurIPSBot/1.0; +http://example.com)'
}

YEARS_TO_SCRAPE = 5  # Number of recent years to scrape

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

# Create a custom SSL context to disable certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Create semaphore for rate limiting
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

def sanitize_filename(name):
    """Sanitize string to be used as a filename."""
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name[:200]  # Limit filename length

async def fetch_html(session, url):
    """Fetch HTML content asynchronously with timeout and rate limiting."""
    async with semaphore:  # Rate limiting
        try:
            async with session.get(url, ssl=ssl_context, headers=HEADERS) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logging.error(f"HTTP {response.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            logging.error(f"Timeout while fetching {url}")
            return None
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
    links = [BASE_URL + a['href'] for a in soup.select("a[href^='/paper_files/paper/']")]
    return sorted(links, reverse=True)  # Most recent years first

async def get_paper_links(session, year_url):
    """Extract paper links for a given year."""
    html = await fetch_html(session, year_url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    return [BASE_URL + a['href'] for a in soup.select("ul.paper-list li a[href$='Abstract-Conference.html']")]

async def process_paper(session, paper_url, pbar):
    """Process a single paper page, extracting metadata and downloading the PDF."""
    try:
        html = await fetch_html(session, paper_url)
        if not html:
            return
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title: try <h2> first; if not found, use the <title> tag as fallback.
        title_tag = soup.find("h2")
        if title_tag:
            title = title_tag.text.strip()
        else:
            head_title = soup.find("title")
            if head_title:
                title = head_title.text.strip()
                # Remove any trailing suffix such as "- Abstract-Conference"
                title = re.sub(r'\s*-\s*Abstract-Conference.*$', '', title)
            else:
                title = "Unknown Title"
        
        # Extract authors from all <i> tags
        authors = ", ".join([i.text.strip() for i in soup.find_all("i")])
        
        # Extract PDF URL using CSS selector
        pdf_link_tag = soup.select_one("a[href$='Paper-Conference.pdf']")
        pdf_url = BASE_URL + pdf_link_tag['href'] if pdf_link_tag else None

        # Extract abstract from <p> tag with class "abstract"
        abstract_tag = soup.find("p", class_="abstract")
        abstract = abstract_tag.text.strip() if abstract_tag else ""

        # Extract year from paper_url using regex
        year_match = re.search(r'/paper_files/paper/(\d{4})/', paper_url)
        year = year_match.group(1) if year_match else ""
        
        metadata = {
            "author": authors,
            "booktitle": "NeurIPS",
            "editor": "",
            "pages": "",
            "publisher": "NeurIPS Foundation",
            "title": title,
            "url": paper_url,
            "volume": year,
            "year": year,
            "abstract": abstract,
            "label": "NeurIPS",
            "pdf_url": pdf_url
        }
        
        if pdf_url:
            await download_pdf(session, pdf_url, title)
        
        save_metadata(metadata)
        pbar.update(1)
        logging.info(f"Processed: {title}")
        
    except Exception as e:
        logging.error(f"Error processing paper {paper_url}: {e}")

async def download_pdf(session, url, title):
    """Download and save a PDF file."""
    safe_title = sanitize_filename(title)
    filename = os.path.join(DOWNLOAD_DIR, f"{safe_title}.pdf")
    
    if os.path.exists(filename):
        logging.info(f"Skipping existing PDF: {filename}")
        return
        
    async with semaphore:  # Rate limiting
        try:
            async with session.get(url, ssl=ssl_context, headers=HEADERS) as response:
                if response.status == 200:
                    with open(filename, "wb") as f:
                        f.write(await response.read())
                    logging.info(f"Downloaded: {title}")
                else:
                    logging.error(f"HTTP {response.status} for PDF {url}")
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")

def save_metadata(metadata):
    """Append paper metadata to the CSV file."""
    fieldnames = ["author", "booktitle", "editor", "pages", "publisher", 
                  "title", "url", "volume", "year", "abstract", "label", "pdf_url"]
    
    file_exists = os.path.isfile(CSV_FILE)
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metadata)
    except Exception as e:
        logging.error(f"Error saving metadata: {e}")

async def main():
    """Main function to orchestrate the scraping."""
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(ssl=ssl_context, limit=MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers=HEADERS
    ) as session:
        # Get all year links
        year_links = await get_year_links(session)
        logging.info(f"Found {len(year_links)} years available.")
        
        # Limit to the last YEARS_TO_SCRAPE years (most recent ones)
        year_links = year_links[:YEARS_TO_SCRAPE]
        logging.info(f"Processing the last {YEARS_TO_SCRAPE} years: {year_links}")
        
        # Process each year in smaller chunks
        for year_url in tqdm(year_links, desc="Processing years"):
            paper_links = await get_paper_links(session, year_url)
            logging.info(f"Found {len(paper_links)} papers for {year_url}")
            
            # Process papers in chunks to avoid memory issues
            chunk_size = 50
            for i in range(0, len(paper_links), chunk_size):
                chunk = paper_links[i:i + chunk_size]
                
                # Create progress bar for current chunk
                with tqdm(
                    total=len(chunk),
                    desc=f"Papers from {year_url.split('/')[-1]} ({i+1}-{i+len(chunk)})"
                ) as pbar:
                    tasks = [process_paper(session, url, pbar) for url in chunk]
                    await asyncio.gather(*tasks)
                
                # Add a small delay between chunks
                await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
