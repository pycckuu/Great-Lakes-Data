graphics.off() 
par("mar") 

library(doParallel)
registerDoParallel(cores=6)

library(EGRET)




species_list <- list()
# species_list['NO3']<- 'Nitrate'
species_list['TP']<- 'Total Phosphorus'
species_list['Cl']<- 'Chloride'
# species_list['SRP']<- 'Soluble Reactive Phosphorus'
# species_list['Si']<- 'Silica'
species_list['NO32']<- 'Nitrate and  Nitrite'

# foreach(idx=1:6) %dopar% {
for (idx in c(1,2,3) ){
  species = names(species_list)[idx]
  for (river in c("Cuyahoga_River", "Maumee_River", "Portage_River", "Raisin_River", "Sandusky_River", "Vermilion_River", "Grand_River_US")){
  # for (river in c("Cattaraugus_Creek_USGS")){
    flow_name = paste("proc_data/Flow_", river, "_data.csv", sep = "")
    sample_name = paste("proc_data/", species, "_", river, "_data.csv", sep = "")
    
    Daily <- readUserDaily('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', flow_name, qUnit = 1)
    Sample <- readUserSample('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', sample_name)
    Sample <- removeDuplicates(Sample)
    
    INFO<-data.frame(0)
    INFO$shortName<- gsub("_", " ", river)
    INFO$param.units<-'mg/l'
    INFO$paramShortName<- species_list[idx]
    INFO$staAbbrev<- '0'
    INFO$constitAbbrev<- species
    INFO$drainSqKm <- 0
    
    eList <- mergeReport(INFO, Daily,Sample)
    eList <- setPA(eList, paStart = 1, paLong = 12)
    eList <- modelEstimation(eList, minNumObs=50)
    
    
    pathLoc = paste("plots/", river,"/", species, "/", river, "_", species, sep="")
    
    multiPlotDataOverview(eList)
    dev.print(pdf, paste(pathLoc, "_multiPlotDataOverview",".pdf", sep = ""))
    
    for (year in 2003:2016) {
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
    saveResults(paste("plots/", river,"/", species, "/", sep=""), eList)
  }
}










