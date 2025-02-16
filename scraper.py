import asyncio
import aiohttp
import aiofiles
import logging
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import certifi
import ssl
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://papers.nips.cc"
OUTPUT_DIR = Path("downloaded_pdfs")
OUTPUT_DIR.mkdir(exist_ok=True)
METADATA_FILE = "papers.csv"

class NeurIPSScraper:
    def __init__(self, start_year: int, end_year: int):
        self.start_year = start_year
        self.end_year = end_year
        self.metadata = []
        
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch HTML content from URL"""
        try:
            async with session.get(url) as response:
                return await response.text()
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""

    async def download_pdf(self, session: aiohttp.ClientSession, url: str, filename: str):
        """Download PDF file"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    filepath = OUTPUT_DIR / filename
                    async with aiofiles.open(filepath, 'wb') as f:
                        await f.write(await response.read())
                    return True
        except Exception as e:
            logger.error(f"Error downloading PDF {url}: {e}")
        return False

    async def parse_paper_page(self, session: aiohttp.ClientSession, paper_url: str):
        """Parse individual paper page and extract metadata"""
        html = await self.fetch_page(session, paper_url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            title = soup.find('h2').text.strip()
            authors = [author.text.strip() for author in soup.find_all('i')]
            pdf_link = soup.select_one("a[href$='Paper-Conference.pdf']")
            
            if pdf_link:
                pdf_url = BASE_URL + pdf_link['href']
                filename = f"{title[:50]}_{authors[0]}.pdf".replace(" ", "_")
                
                if await self.download_pdf(session, pdf_url, filename):
                    return {
                        'title': title,
                        'authors': '; '.join(authors),
                        'year': paper_url.split('/')[-2],
                        'pdf_url': pdf_url,
                        'local_pdf': str(OUTPUT_DIR / filename)
                    }
        except Exception as e:
            logger.error(f"Error parsing paper page {paper_url}: {e}")
        
        return None

    async def process_year(self, session: aiohttp.ClientSession, year: int):
        """Process all papers for a given year"""
        year_url = f"{BASE_URL}/paper_files/paper/{year}"
        html = await self.fetch_page(session, year_url)
        
        if not html:
            return

        soup = BeautifulSoup(html, 'html.parser')
        paper_links = soup.select("ul.paper-list li a[href$='Abstract-Conference.html']")
        
        tasks = []
        for link in paper_links:
            paper_url = BASE_URL + link['href']
            tasks.append(self.parse_paper_page(session, paper_url))
        
        results = await asyncio.gather(*tasks)
        self.metadata.extend([r for r in results if r])

    async def run(self):
        """Main execution method"""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for year in range(self.start_year, self.end_year + 1):
                tasks.append(self.process_year(session, year))
            
            with tqdm(total=len(tasks), desc="Processing years") as pbar:
                for task in asyncio.as_completed(tasks):
                    await task
                    pbar.update(1)
        
        # Save metadata to CSV
        df = pd.DataFrame(self.metadata)
        df.to_csv(METADATA_FILE, index=False)
        logger.info(f"Saved metadata for {len(self.metadata)} papers to {METADATA_FILE}")
        
def main():
    # Example usage
    start_year = 2018
    end_year = 2023
    
    scraper = NeurIPSScraper(start_year, end_year)
    asyncio.run(scraper.run())

if __name__ == "__main__":
    main()