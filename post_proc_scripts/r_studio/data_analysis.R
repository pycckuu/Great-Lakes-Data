graphics.off() 
par("mar") 


library(EGRET)


INFO<-data.frame(0)
INFO$shortName<- 'Cuyahoga River'
INFO$param.units<-'mg/l'
INFO$paramShortName<- 'Total Phosphorus'
INFO$staAbbrev<- '132112'
INFO$constitAbbrev<- 'TP'
INFO$drainSqKm <- 0
Daily <- readUserDaily('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', 'Daily_s.csv', qUnit = 2)
Sample <- readUserSample('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', 'Sample_s.csv')
Sample <- removeDuplicates(Sample)
# Daily <- subset(Daily, Date > as.Date("1995-01-01") )
# Sample <- subset(Sample, Date > as.Date("1995-01-01") )
eList <- mergeReport(INFO, Daily,Sample)
eList <- setPA(eList, paStart = 1, paLong = 12)
eList <- modelEstimation(eList)



############################
# Check sample data:
boxConcMonth(eList)
boxQTwice(eList)
plotConcTime(eList)
plotConcQ(eList)
multiPlotDataOverview(eList)
############################


############################
#Check model results:

#Require Sample + INFO:
plotConcTimeDaily(eList, yearStart = 1995, yearEnd = 1996)
plotFluxTimeDaily(eList, yearStart = 1995, yearEnd = 1996)
plotConcPred(eList)
plotFluxPred(eList)
plotResidPred(eList)
plotResidQ(eList)
plotResidTime(eList)
boxResidMonth(eList)
boxConcThree(eList)

#Require Daily + INFO:
plotConcHist(eList)
plotFluxHist(eList)

# Multi-line plots:
date1 <- "2000-09-01"
date2 <- "2005-09-01"
date3 <- "2009-09-01"
qBottom<-1
qTop<-100
plotConcQSmooth(eList, date1, date2, date3, qBottom, qTop, 
                concMax=5,qUnit=2)
q1 <- 10
q2 <- 25
q3 <- 75
centerDate <- "07-01"
yearEnd <- 2009
yearStart <- 2000
plotConcTimeSmooth(eList, q1, q2, q3, centerDate, yearStart, yearEnd, qUnit = 2)

# Multi-plots:
fluxBiasMulti(eList)

#Contour plots:
clevel<-seq(0,1,0.1)
maxDiff<-0.8
yearStart <- 2000
yearEnd <- 2010

plotContours(eList, yearStart,yearEnd,qBottom=10,qTop=100, 
             contourLevels = clevel,qUnit=2)
plotDiffContours(eList, yearStart,yearEnd,
                 qBottom=10,qTop=100,maxDiff,qUnit=2)


plotConcQSmooth(eList,date1, date2,date3,qBottom, qTop,concMax=1,qUnit=2)
