import pandas as pd
import requests
import time

# Credentials
token = "d8255a86f2ae0b3342c288641a9e130486d7947b"


# Helper functions
def station_dict_to_df(dict):
    """
    Convert station dictionary from API into a data frame
    :param dict: a dictionary containing data from each station
    :return: dataframe including each station's information
    """
    df = pd.DataFrame(columns=['uid', 'lat', 'lon', 'name', 'aqi'])
    for station_data in dict:
        station_series = {'uid': [station_data['uid']],
                          'lat': [station_data['lat']],
                          'lon': [station_data['lon']],
                          'name': [station_data['station']['name']],
                          'aqi': [station_data['aqi']]}
        df = pd.concat([df, pd.DataFrame(data=station_series)], axis=0)
    return df


# Set lat/lon bounding boxes to obtain data from all stations
latitudes = list(range(90, -90, -4))
longitudes = list(range(-200, 200, 4))

boxes = []
for lat in latitudes:
    for lon in longitudes:
        boxes.append((lat, lon, lat-4, lon+4))

# Set up data frame for station data
stations = pd.DataFrame(columns=['uid', 'lat', 'lon', 'name', 'aqi'])

# Obtain station data for each bounding box
print('[INFO] Obtaining stations within lat-lon limits...')
for i, box in enumerate(boxes):
    # Request and get data
    lat1, lon1, lat2, lon2 = box
    r = requests.get(
        'https://api.waqi.info/map/bounds/?latlng={},{},{},{}&token={}'.format(
            lat1, lon1, lat2, lon2, token))
    stations_dict = r.json()

    # Reformat into DataFrame and append to stations DataFrame
    if stations_dict['status'] == 'ok':
        print('[INFO] Getting stations in box {}/{}'.format(i, len(boxes)))
        stations = stations.append(station_dict_to_df(stations_dict['data']),
                        ignore_index=True)
        print('[INFO] Total stations processed: {}'.format(stations.shape[0]))

        # Pace requests
        time.sleep(0.5)
    else:
        print("[INFO] Request error for box {},{},{},{}".format(
            lat1, lon1, lat2, lon2))


# Clean data
stations.drop_duplicates(subset=['uid'], keep='first', inplace=True, )
stations = stations.set_index('uid')
stations.to_csv('../01_Data/01_Carbon_emissions/WAQI_station_data_aqi_Jan_21.csv')
