import os
import requests
from dotenv import load_dotenv

load_dotenv()

NASA_ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"
NASA_ADS_API_KEY = os.getenv("NASA_ADS_API_KEY")

def search_nasa_ads(query):
    headers = {
        "Authorization": f"Bearer {NASA_ADS_API_KEY}"
    }
    params = {
        "q": query,
        "fl": "title,author,abstract",
        "rows": 5
    }
    print(f"Querying NASA ADS with query: {query}")
    response = requests.get(NASA_ADS_API_URL, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error querying NASA ADS: {response.status_code}")
        return None