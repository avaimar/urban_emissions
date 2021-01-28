library(data.table)
library(stars)


# 1. Forecast: Reporting area --------------------------------
dat <- fread('https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/reportingarea.dat')

# Frequency: daily
# Count number of unique sites: only 788
length(unique(dat$V8))

# Seem to almost only be in the US
length(unique(dat$V9))
sum(dat$V9 == "")

# 2. Reporting area info --------------------------------
dat <- fread('https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/reporting_area_locations_V2.dat',
             col.names = c('Reporting_area', 'State_code', 'Country_code', 'Forecasts',
                           'Action_day_name', 'Lat', 'Lon', 'GMT_offset', 'Daylight_savings',
                           'std_time_zone_label', 'DLS_time_zone_label', 'TWC_code', 'USA_today',
                           'forecast_source'))

# Frequency: last updated 2012
# Number of reporting areas: 917
length(unique(dat$Reporting_area))

# All US? 41 countries
unique(dat$Country_code)

# 3. Hourly data values --------------------------------
dat <- fread('https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/HourlyData_2020010100.dat')

# Frequency: hourly
# Number of unique sites: 3496, but varies a lot by pollutant. Eg. 1796 for NO2
length(unique(dat$V3))
dat[, .(N = .N), by = .(V6)]

# 4. Monitoring site info --------------------------------
dat <- fread('https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/monitoring_site_locations.dat',
             col.names= c('AQSID', 'Parameter_name', 'Site_code', 'Site_name', 'Status',
                         'Agency_ID', 'Agency_name', 'EPA_region', 'lat', 'lon', 'elevation',
                         'GMT_offset', 'country_code', 'blank', 'blank2', 'MSA_code', 'MSA_name',
                         'State_code', 'state_name', 'county_code', 'county_name', 'blank3',
                         'blank4'))

# Frequency: 
# Number of sites: 2951
length(unique(dat$AQSID))

# Number of counties: 1138
length(unique(dat$county_code))

# 5. AirNow mapping information --------------------------------
# "Gridded data files [..] Each file contains a grid that covers the contiguous US and contains
# AQI observations for a given hour
# Frequency: daily, current hour
# One file per pollutant
# Note: grid resolution is ~8km

dat <- read_stars("US-191230-max_pm25.grib2")
plot(dat)

# 6. 10-day look-back --------------------------------
dat <- fread('https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/ten_day_lookback.dat')
# Frequency: sub-hourly

# 674 unique sites
length(unique(dat$V2))

# 10 days
unique(dat$V1)

# 7. Daily data values --------------------------------
dat <- fread('https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/daily_data_v2.dat')

# 2727 unique sites
length(unique(dat$V2))

# Pollutants
unique(dat$V4)

# Max. unique values is 1119 for SO2-24HR
dat[, .(N = .N), by = .(V4)]

# 8. Hourly AQ Obs --------------------------------
# "Site focused data file containing corresponding reporting area, AQI values, raw concentrations
# and site information"
dat <- fread('https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/2020/20200101/HourlyAQObs_2020010100.dat')

# Unique sites: 2954
length(unique(dat$AQSID))

# We have many missings even though number of unique sites is 3k
colSums(is.na(dat))

# 9. Zip code mappings to reporting areas
# Note: this is only for the US
dat <- fread('https://files.airnowtech.org/airnow/today/cityzipcodes.csv')

# Number of zip codes: 16187
length(unique(dat$Zipcode))

# Locations
unique(dat$State)

# Number of sites
length(unique(dat$City))


