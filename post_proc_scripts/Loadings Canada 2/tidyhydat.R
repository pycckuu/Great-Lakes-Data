
# download_hydat()
library(tidyhydat)


station = '02GB001'
flo = hy_daily_flows(station_number = station, start_date = '1996-01-01', end_date = '2018-01-01')

plot(flo$Date, flo$Value)
