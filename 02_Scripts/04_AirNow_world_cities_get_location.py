import pandas as pd
import requests


# Authentication
from config import google_api_key

# Load city data
cities_data = pd.read_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_cities_2020_avg.csv')

# Get unique city names
cities = cities_data['name'].unique()

# Get lat lon
base_url = 'https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}'
locations = []
for city in cities:
    # Transform city string and request geolocation
    city_str = city.replace(' ', '%20')
    try:
        r = requests.get(base_url.format(city_str, google_api_key))
    except:
        print("[INFO] Error obtaining location for {}".format(city))

    # Extract latitude and longitude
    r = r.json()
    if r['status'] == 'OK':
        lat = r['results'][0]['geometry']['location']['lat']
        lon = r['results'][0]['geometry']['location']['lng']
        locations.append((city, lat, lon))
        if len(r['results']) > 1:
            print('[WARNING] Multiple matches for {}'.format(city))
    else:
        print("[INFO] Error obtaining location for {}".format(city))

# Merge city data and locations
city_table = pd.DataFrame(locations)
city_table.columns = ['name', 'lat', 'lon']
cities_data = pd.merge(cities_data, city_table, how='left',
                       on='name')

# Manually get locations with multiple matches

