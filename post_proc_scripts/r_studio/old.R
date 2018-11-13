graphics.off() 
par("mar") 
# par(mar=c(1,1,1,1))

# decimal date lubridate library
library(lubridate)
# Moving averages
library(TTR)
# na.approx function
library(zoo)
library(EGRET)
library(dplyr)

Cuyahoga_River_data = read.table(file='../../measurements/ncwqr.org/Cuyahoga_River_data.csv',header=T, sep=",", stringsAsFactors = F, na.strings = "-9")
Cuyahoga_River_data[Cuyahoga_River_data < 0] <- NA
Cuyahoga_River_data$Date <- as.Date(Cuyahoga_River_data[,c("Datetime..date.and.time.of.sample.collection.")], format="%m-%d-%Y %H:%M")
Cuyahoga_River_data<-aggregate(Cuyahoga_River_data, by=list(Cuyahoga_River_data$Date), mean)

monnb <- function(d) { lt <- as.POSIXlt(as.Date(d, origin="1900-01-01")); lt$year*12 + lt$mon }
mondf <- function(d1, d2) { monnb(d2) - monnb(d1) }

Daily<-setNames(data.frame(Cuyahoga_River_data[,c("Date")], na.approx(as.numeric(Cuyahoga_River_data[,c("Flow..CFS")]*0.0283168))), c("Date", "Q"))

ts <- seq.POSIXt(as.POSIXlt(Cuyahoga_River_data[,c("Date")][1]), as.POSIXlt(tail(Cuyahoga_River_data[,c("Date")], 1)), by="day")
df = data.frame(Date=as.Date(ts))
Daily = full_join(Daily,df)
Daily <-arrange(Daily, Date)

Daily$Q <- na.approx(Daily$Q)
Daily$Julian <- as.numeric(Daily$Date -  as.Date("1850/01/01", format="%Y/%m/%d"))
Daily$Month <- month(Daily$Date)
Daily$Day <- yday(Daily$Date)
Daily$DecYear <- decimal_date(Daily$Date)
Daily$MonthSeq <- mondf(as.Date("1850-01-01"), Daily$Date)
Daily$Qualifier<-'A'
Daily$i <- as.numeric(rownames(Daily))
Daily$LogQ <- log(Daily$Q)
Daily$Q7 <- SMA(Daily$Q, n=7)
Daily$Q30 <- SMA(Daily$Q, n=30)

Sample<-setNames(data.frame(Cuyahoga_River_data[,c("Date")], na.approx(as.numeric(Cuyahoga_River_data[,c("Flow..CFS")]*0.0283168))), c("Date", "Q"))
Sample$ConcLow<-as.numeric(Cuyahoga_River_data[,c("TP..mg.L.as.P")])

Sample<-Sample[complete.cases(Sample), ]
summary(Sample)

Sample$ConcHigh<-Sample$ConcLow
Sample$ConcAve<-Sample$ConcLow
Sample$Uncen<-1

Sample$DecYear <- decimal_date(Sample$Date)
Sample$Day <- yday(Sample$Date)
Sample$Month <- month(Sample$Date)
Sample$MonthSeq <- mondf(as.Date("1850-01-01"), Sample$Date)
Sample$LogQ <- log(Sample$Q)
Sample$Julian <- as.numeric(Sample$Date -  as.Date("1850/01/01", format="%Y/%m/%d"))
# Sample$SinDY <-  sin(2*pi*decimal_date(Sample$Date))
# Sample$CosDY<- cos(2*pi*decimal_date(Sample$Date))

Sample <- removeDuplicates(Sample)

data<-ChoptankFlow


# summary(Daily)

INFO <- readNWISInfo("", "")
INFO<-data.frame(0)
INFO$shortName<- 'Cuyahoga River'
INFO$paramShortName<- 'Total Phosphorus'
INFO$staAbbrev<- '132112'
INFO$constitAbbrev<- 'TP'
INFO$drainSqKm <- 0
# INFO<-setNames(data.frame('Cuyahoga River','Total Phosphorus','CR','TP', NA), c('shortName','paramShortName','staAbbrev','constitAbbrev','drainSqKm'))

eList <- mergeReport(INFO, Daily,Sample)

eList <- setPA(eList, paStart = 1, paLong = 12, window = 20)


# write.csv(Daily,'Daily.csv')
# write.csv(Sample,'Sample.csv')

Daily <- readUserDaily('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', 'Daily_s.csv', qUnit = 2)
Sample <- readUserSample('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', 'Sample_s.csv')



boxQTwice(eList)

plot(Sample$DecYear, Sample$Q, 'l')



plotFlowSingle(eList, istat = 2, qUnit = 2)


modelEstimation(eList,minNumObs=100, minNumUncen=100, edgeAdjust=TRUE)
