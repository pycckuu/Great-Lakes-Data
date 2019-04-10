library(tidyhydat)



station = '02GG011'
flo = hy_daily_levels(station_number = station, start_date = '1900-01-01', end_date = '2018-01-01')


flo_an = aggregate(flo, list(flo$Year), mean)

write.csv(flo,'stclair.csv')

plot(flo_an$Group.1, flo_an$Value)

