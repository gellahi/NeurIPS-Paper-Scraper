# NeurIPS Paper Scraper

An asynchronous scraper for downloading NeurIPS conference papers and metadata.

## Setup

1. Install dependencies:
```sh
poetry install
```
2. Run the scraper:
```sh
poetry run python scraper.py
```
3. Features
- Asynchronous paper downloading
- Metadata extraction
- Progress tracking
- CSV export


To get started, you can create the project structure by running these commands:

```sh
mkdir -p downloaded_pdfs
touch scraper.py
poetry install