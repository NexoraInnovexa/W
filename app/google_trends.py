

from pytrends.request import TrendReq

def get_google_trends(idea):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([idea], cat=0, timeframe='today 12-m', geo='', gprop='')
    trends_data = pytrends.interest_over_time()
    
    if trends_data.empty:
        return "No data found for this search term."
    
    return trends_data.tail(1).to_dict()  # Return the latest trend data
