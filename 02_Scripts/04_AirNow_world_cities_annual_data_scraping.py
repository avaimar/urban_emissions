import pandas as pd


# Set up selected year, months, days, hour
years = ["2020"]
months = [str(num).zfill(2) for num in list(range(1, 13))]
all_days = [str(num).zfill(2) for num in list(range(1, 32))]
hours = [str(num).zfill(2) for num in list(range(24))]

# Base URL
base_url = "https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow"

# Set up data containers
hourly_data = pd.DataFrame()

# Loop over all years, months, days, hours
for year in years:
    print('[INFO] Loading data for the year {}'.format(year))
    tab_year = pd.DataFrame()
    for month in months:
        print('[INFO] Loading data for month {}, year {}'.format(month, year))
        # Pick the correct number of days for the month
        if month in ["04", "06", "09", "11"]:
            days = all_days[:-1]
        elif month == "02":
            days = all_days[:-3]
        else:
            days = all_days
        # Create a new month table
        tab_month = pd.DataFrame()
        for day in days:
            print('[INFO] Loading data for day {}, month {}, year {}'.format(
                day, month, year))
            # Create a day table
            tab_day = pd.DataFrame()
            for hour in hours:
                # Get HourlyData query
                query = "{}/{}/{}{}{}/HourlyData_{}{}{}{}.dat".format(
                    base_url,
                    year,
                    year, month, day,
                    year, month, day, hour
                )

                # Read data and append to day table
                try:
                    tab_hour = pd.read_table(query, sep="|", header=0,
                                             names=['date', 'time', 'id', 'name',
                                                    'XX', 'type', 'measurement',
                                                    'value', 'institution'])
                    tab_day = pd.concat([tab_day, tab_hour], axis=0)

                    # Print missing value alert
                    if tab_hour.isna().sum(axis=0)['value'] > 0:
                        print('[WARNING] Found missing value data in hourly table for',
                              'hour {}, day {}, month {}, year {}'.format(
                                  hour, day, month, year))

                except:
                    print("[INFO] Error loading dataset {}/{}/{}/{}".format(
                        year, month, day, hour))

            # Compute daily average and append to month table
            tab_day = tab_day.drop('time', axis=1)
            tab_day = tab_day \
                .groupby(['date', 'id', 'name', 'XX', 'type',
                          'measurement', 'institution'], as_index=False) \
                .mean()
            tab_month = pd.concat([tab_month, tab_day], axis=0)
        # Compute monthly average and append to month table
        tab_month = tab_month.drop('date', axis=1)
        tab_month = tab_month \
            .groupby(['id', 'name', 'XX', 'type',
                      'measurement', 'institution'], as_index=False) \
            .mean()
        tab_year = pd.concat([tab_year, tab_month], axis=0)

    # Compute annual average and append to hourly_data table
    tab_year = tab_year \
        .groupby(['id', 'name', 'XX', 'type',
                  'measurement', 'institution'], as_index=False) \
        .mean()
    tab_year['year'] = year
    hourly_data = pd.concat([hourly_data, tab_year], axis=0)

print('[INFO] Data scraping completed.')
hourly_data.to_csv(
    '../01_Data/01_Carbon_emissions/AirNow/World_cities_2020_avg_raw.csv',
    index=False)
