import os
import requests
from dotenv import load_dotenv

load_dotenv()

def query_nasa_ads(query):
    api_key = os.getenv('NASA_ADS_API_KEY')
    if not api_key:
        raise ValueError("NASA_ADS_API_KEY environment variable not set")

    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    params = {
        'q': query,
        'fl': 'title,author,year',
        'rows': 10
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()