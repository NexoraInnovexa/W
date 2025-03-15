# crunchbase.py

import requests

def get_competitor_data(company_name):
    api_key = "your_crunchbase_api_key"
    url = f"https://api.crunchbase.com/v3.1/organizations?user_key={api_key}&query={company_name}"
    
    response = requests.get(url)
    data = response.json()
    
    if 'data' not in data:
        return "No data found for this company."
    
    company_data = data['data']['items'][0]  # Get the first result
    company_info = {
        'name': company_data['name'],
        'funding': company_data['funding_rounds'],
        'valuation': company_data['valuation'],
        'industry': company_data['industry'],
    }
    
    return company_info
