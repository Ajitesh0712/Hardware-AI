import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://real-time-amazon-data.p.rapidapi.com/search"

headers = {
    "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
    "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
}

params = {
    "query": "laptop",
    "page": "1",
    "country": "US"
}

response = requests.get(url, headers=headers, params=params)
print("Status Code:", response.status_code)
print("Response:", response.text[:500])