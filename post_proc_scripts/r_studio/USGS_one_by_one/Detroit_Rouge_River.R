graphics.off() 
par("mar") 



library(EGRET)
river = "Rouge_River_US_Detroit"
############################
# Gather discharge data:
siteID <- "04166500"
startDate <- "1980-01-01" #Gets earliest date
endDate <- "2018-10-30"
# Gather sample data:
Daily <- readNWISDaily(siteID,"00060",startDate,endDate)



species_list <- list()
species_list['TP']<- 'Total Phosphorus'
species_list['NO32']<- 'Nitrate and Nitrite'
species_list['NO3']<- 'Total Nitrate'
species_list['Cl']<- 'Chloride'
species_list['PO4']<- 'Total Ortho Phosphate'
# species_list['Si']<- 'Silica'


for (idx in c(1, 2, 3, 4, 5) ){
  short_name = names(species_list)[idx]
  long_name = species_list[idx]
  
  sample_name=paste(short_name, "_MIDEQ-820070.csv", sep="")
  Sample <- readUserSample('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/proc_data/', sample_name)
  
  INFO<-data.frame(0)
  INFO$shortName<- gsub("_", " ", river)
  INFO$param.units<-'mg/l'
  INFO$paramShortName<- long_name
  INFO$staAbbrev<- '0'
  INFO$constitAbbrev<- short_name
  INFO$drainSqKm <- 0
  
  
  eList <- mergeReport(INFO, Daily,Sample)
  eList <- setPA(eList, paStart = 1, paLong = 12)
  eList <- modelEstimation(eList, minNumUncen=10, minNumObs=10)
  
  
  pathLoc = paste("plots/", river,"/", short_name, "/", river, "_", short_name, sep="")
  
  multiPlotDataOverview(eList)
  dev.print(pdf, paste(pathLoc, "_multiPlotDataOverview",".pdf", sep = ""))
  
  for (year in 2003:2018) {
    plotConcTimeDaily(eList, yearStart = year, yearEnd = year+1)  
    dev.print(pdf, paste(pathLoc, "_", year, "_conc.pdf", sep = ""))
    plotFluxTimeDaily(eList, yearStart = year, yearEnd = year+1)
    dev.print(pdf, paste(pathLoc, "_", year, "_flux.pdf", sep = ""))
  }
  
  plotConcPred(eList)
  dev.print(pdf, paste(pathLoc, "_plotConcPred.pdf", sep = ""))
  plotFluxPred(eList)
  dev.print(pdf, paste(pathLoc, "_plotFluxPred.pdf", sep = ""))
  plotResidPred(eList)
  dev.print(pdf, paste(pathLoc, "_plotResidPred.pdf", sep = ""))
  plotResidQ(eList)
  dev.print(pdf, paste(pathLoc, "_plotResidQ.pdf", sep = ""))
  plotResidTime(eList)
  dev.print(pdf, paste(pathLoc, "_plotResidTime.pdf", sep = ""))
  boxResidMonth(eList)
  dev.print(pdf, paste(pathLoc, "_boxResidMonth.pdf", sep = ""))
  boxConcThree(eList)
  dev.print(pdf, paste(pathLoc, "_boxConcThree.pdf", sep = ""))
  
  plotConcHist(eList)
  dev.print(pdf, paste(pathLoc, "_plotConcHist",".pdf", sep = ""))
  plotFluxHist(eList)
  dev.print(pdf, paste(pathLoc, "_plotFluxHist",".pdf", sep = ""))
  saveResults(paste("plots/", river,"/", short_name, "/", sep=""), eList)
  
  
daily_name = paste("plots/", river,"/", short_name, "/", river, "_", short_name, '_Daily.csv',sep="")
sample_name = paste("plots/", river,"/", short_name, "/", river, "_", short_name, '_Sample.csv',sep="")
write.csv(eList[["Daily"]], daily_name)
write.csv(eList[["Sample"]], sample_name)

}








