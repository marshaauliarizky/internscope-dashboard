import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def scrape_glints(keyword="data analyst", location="Indonesia", max_pages=5):
    jobs = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    for page in range(1, max_pages + 1):
        url = f"https://glints.com/id/opportunities/jobs/explore?keyword={keyword}&country=ID&locationName={location}&page={page}"
        print(f"Scraping page {page}...")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            job_cards = soup.find_all("div", class_="CompactOpportunityCardsc__CardWrapper")
            
            for card in job_cards:
                title = card.find("h2")
                company = card.find("span", class_="CompactOpportunityCardsc__CompanyName")
                location_tag = card.find("span", class_="location")
                
                jobs.append({
                    "title": title.text.strip() if title else "N/A",
                    "company": company.text.strip() if company else "N/A",
                    "location": location_tag.text.strip() if location_tag else "N/A",
                    "keyword": keyword
                })
            
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            print(f"Error page {page}: {e}")
            continue
    
    return pd.DataFrame(jobs)

if __name__ == "__main__":
    keywords = ["data analyst", "data science", "software developer", "machine learning"]
    all_jobs = []
    
    for kw in keywords:
        print(f"\n🔍 Scraping: {kw}")
        df = scrape_glints(keyword=kw, max_pages=3)
        all_jobs.append(df)
        time.sleep(2)
    
    final_df = pd.concat(all_jobs, ignore_index=True)
    final_df.to_csv("data/jobs_raw.csv", index=False)
    print(f"\n✅ Done! Total: {len(final_df)} jobs saved to data/jobs_raw.csv")