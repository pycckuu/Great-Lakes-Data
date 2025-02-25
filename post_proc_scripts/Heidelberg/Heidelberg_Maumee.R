graphics.off() 
par("mar") 

library(doParallel)
registerDoParallel(cores=6)

library(EGRET)

# startDate <- "1996-01-01"
# endDate <- "2018-01-01"

species_list <- list()
# species_list['TP']<- 'Total Phosphorus'
# species_list['SRP']<- 'Soluble Reactive Phosphorus'
# species_list['NO3']<- 'Nitrate'
species_list['NO32']<- 'Nitrate and  Nitrite'
species_list['Cl']<- 'Chloride'
species_list['Si']<- 'Silica'

# foreach(idx=1:6) %dopar% {
for (idx in c(1,2,3,4,5,6) ){
  species = names(species_list)[idx]
  river = "Maumee_River"
    flow_name = paste("proc_data/Flow_", river, "_data.csv", sep = "")
    sample_name = paste("proc_data/", species, "_", river, "_data.csv", sep = "")
    
    Daily <- readUserDaily('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/Heidelberg/', flow_name, qUnit = 1)
    Sample <- readUserSample('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/Heidelberg/', sample_name)
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
    eList <- modelEstimation(eList)
    
    
    pathLoc = paste("/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/Heidelberg/plots/", river,"/", species, "/", river, "_", species, sep="")
    
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
    
    
    daily_name = paste(pathLoc, river, "_", species, '_Daily.csv',sep="")
    sample_name = paste(pathLoc, river, "_", species, '_Sample.csv',sep="")
    write.csv(eList[["Daily"]], daily_name)
    write.csv(eList[["Sample"]], sample_name)
}










