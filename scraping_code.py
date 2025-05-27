import requests
import pandas as pd
from bs4 import BeautifulSoup
from scholarly import scholarly
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from datetime import datetime

class PaperScraper:
    def __init__(self, start_year=2023, end_year=2025):
        self.start_year = start_year
        self.end_year = end_year
        self.papers = []
        
    def setup_selenium(self):
        """Initialize headless Firefox browser"""
        firefox_options = webdriver.FirefoxOptions()
        firefox_options.add_argument("--headless")
        firefox_options.add_argument("--disable-gpu")
        firefox_options.add_argument("--no-sandbox")
        return webdriver.Firefox(options=firefox_options)
    
    def search_google_scholar(self, query="AI geotechnical engineering", max_papers=100):
        """Scrape papers from Google Scholar"""
        print(f"Searching Google Scholar for: {query}")
        
        # Use scholarly for Google Scholar scraping
        search_query = scholarly.search_pubs(query, year_low=self.start_year, year_high=self.end_year)
        count = 0
        print(f"Found {len(list(search_query))} papers.")
        for paper in search_query:
            if count >= max_papers:
                break
                
            try:
                # Get complete publication data
                pub = scholarly.fill(paper)
                
                # Extract year (if available)
                year = pub.get('bib', {}).get('pub_year')
                if year and self.start_year <= int(year) <= self.end_year:
                    # Extract data
                    title = pub.get('bib', {}).get('title', '')
                    authors = ', '.join(pub.get('bib', {}).get('author', '').split(' and '))
                    journal = pub.get('bib', {}).get('journal', '')
                    
                    self.papers.append({
                        'Title': title,
                        'Authors': authors,
                        'Year': year,
                        'Journal': 'Y' if journal else 'N',
                        'URL': pub.get('pub_url', ''),
                        'Abstract': pub.get('bib', {}).get('abstract', '')
                    })
                    count += 1
                    print(f"Added paper: {title}")
                
            except Exception as e:
                print(f"Error processing paper: {str(e)}")
            
            # Avoid being blocked
            time.sleep(2)
            
    def search_arxiv(self, query="AI geotechnical engineering", max_papers=50):
        """Scrape papers from arXiv"""
        print(f"Searching arXiv for: {query}")
        
        # Format query for arXiv API
        formatted_query = '+AND+'.join(query.split())
        url = f"http://export.arxiv.org/api/query?search_query=all:{formatted_query}&start=0&max_results={max_papers}&sortBy=submittedDate&sortOrder=descending"
        
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'xml')
            entries = soup.find_all('entry')
            
            for entry in entries:
                try:
                    # Extract publication date
                    published = entry.published.text
                    year = int(published[:4])
                    
                    if self.start_year <= year <= self.end_year:
                        # Extract data
                        title = entry.title.text.strip()
                        
                        # Extract authors
                        author_tags = entry.find_all('author')
                        authors = ', '.join([author.find('name').text for author in author_tags])
                        
                        # Extract abstract and URL
                        abstract = entry.summary.text.strip()
                        url = entry.find('link', {'rel': 'alternate'}).get('href', '')
                        # <link title="pdf" href="http://arxiv.org/pdf/2501.14186v1" rel="related" type="application/pdf"/> get the pdf link
                        pdf_link = entry.find('link', {'title': 'pdf'}).get('href', '')
                        # download the pdf
                        pdf_response = requests.get(pdf_link)
                        # remove special characters from title
                        title = re.sub(r'[\\/*?:"<>|]', '', title)
                        # if \n the title, remove it
                        title = title.replace('\n', '')
                        with open(f"{title}.pdf", "wb") as f:
                            f.write(pdf_response.content)
                        
                        self.papers.append({
                            'Title': title,
                            'Authors': authors,
                            'Year': year,
                            'Journal': 'N',  # arXiv is not a journal
                            'URL': url,
                            'Abstract': abstract,
                            "Document": f"{title}.pdf"
                        })
                        print(f"Added paper: {title}")
                        
                except Exception as e:
                    print(f"Error processing arXiv entry: {str(e)}")
    


    def search_sciencedirect(self, query="AI geotechnical engineering", max_results=50, start=0, api_key=None):
        """Fetch papers from ScienceDirect using Elsevier API"""
        print(f"Searching ScienceDirect for: {query}")

        # Replace with your actual API key and label
        

        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json"
        }

        base_url = "https://api.elsevier.com/content/search/scopus"
        params = {
            "query": query,
            "count": max_results,
            "start": start,
        }

        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            for result in data.get("search-results", {}).get("entry", []):
                try:
                    title = result.get("dc:title", "")
                    open_access = result.get("openaccess", "N")
                    if bool(int(open_access)) == False:
                        continue
                    authors = result.get("authors", "")
                    year = result.get("prism:coverDate", "")[:4]
                    url = result.get("link", [{}])[0].get("@href", "")
                    doi = result.get("prism:doi", "")
                    # Crossref API endpoint for works
                    url = f"https://api.crossref.org/works/{doi}"
                    headers = {
                        "User-Agent": "python-get-abstract/1.0 (mailto:your-email@example.com)"
                    }
                    resp = requests.get(url, headers=headers)
                    resp.raise_for_status()

                    message = resp.json().get("message", {})
                    document = message.get("link")
                    if not document:
                        continue
                    else:
                        # get the url of the first link
                        url = message.get("link")[0].get("URL")
                        # download the pdf
                        pdf_response = requests.get(url)
                        with open(f"{title}.pdf", "wb") as f:
                            f.write(pdf_response.content)

                    if year and self.start_year <= int(year) <= self.end_year:
                        self.papers.append({
                            'Title': title,
                            'Authors': authors,
                            'Year': year,
                            'Journal': 'Y',
                            'URL': url,
                            'Abstract': '',
                            "Document": f"{title}.pdf",
                        })
                        print(f"Added paper: {title}")

                except Exception as e:
                    print(f"Error processing ScienceDirect paper: {str(e)}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Elsevier API: {str(e)}")
    
    def export_to_csv(self, filename="geotechnical_ai_papers.csv"):
        """Export collected papers to CSV"""
        df = pd.DataFrame(self.papers)
        df.to_csv(filename, index=False)
        print(f"Exported {len(self.papers)} papers to {filename}")
        return filename

    def get_abstract_from_doi(doi: str) -> str:
        """
        Fetches the abstract for a given DOI via Crossref.
        Returns the abstract as plain text (or None if not available).
        """
        # Crossref API endpoint for works
        url = f"https://api.crossref.org/works/{doi}"
        headers = {
            "User-Agent": "python-get-abstract/1.0 (mailto:your-email@example.com)"
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        message = resp.json().get("message", {})
        raw_abstract = message.get("abstract")
        if not raw_abstract:
            return None

        # Crossref abstracts are often in JATS XML; strip tags to get plain text
        soup = BeautifulSoup(raw_abstract, "lxml")
        return soup.get_text(separator=" ", strip=True)

# Example usage
if __name__ == "__main__":
    scraper = PaperScraper(start_year=2023, end_year=2025)
    query = "AI geotechnical engineering "
    api_key = "YOUR_ELSEVIER_API_KEY"  # Replace with your actual API key
    # Search different sources
    #scraper.search_google_scholar(query=query, max_papers=50)
    scraper.search_arxiv(query=query, max_papers=30)
    for start in range(0, 200, 10):
        scraper.search_sciencedirect(query=query, max_results=10, start=start, api_key=api_key)
    
    # Export results
    csv_file = scraper.export_to_csv()
    print(f"Total papers collected: {len(scraper.papers)}")
