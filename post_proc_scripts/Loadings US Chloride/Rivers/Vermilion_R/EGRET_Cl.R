rm(list=ls())

graphics.off() 
par("mar") 

# library(doParallel)
# registerDoParallel(cores=6)

library(EGRET)

# startDate <- "1996-01-01"
# endDate <- "2018-01-01"

species_list <- list()
species_list['Cl']<- 'Chloride'
# species_list['SRP']<- 'Soluble Reactive Phosphorus'
# species_list['NO32']<- 'Nitrate'
# # species_list['TN']<- 'Total Nitrogen'
# species_list['Cl']<- 'Chloride'
# species_list['Si']<- 'Silica'

# foreach(idx=1:5) %dopar% {
# for(idx in c(1,2,3,4,5)) {
idx=1
  species = names(species_list)[idx]
  river = "Vermilion_River"
  file_loc = paste("/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/Loadings US Chloride/Rivers/Vermilion_R/", sep="")
    Daily <- readUserDaily(file_loc, "Flow_Vermilion_R_CFS.csv", qUnit = 1)
    Sample <- readUserSample(file_loc, "Cl_Vermilion_River_data.csv")
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
    eList <- modelEstimation(eList, minNumUncen = 20, minNumObs=20)
    
    
    pathLoc = paste(file_loc, "EGRET Result/", species, "/", river, "_", species, sep="")
    
    multiPlotDataOverview(eList)
    dev.print(pdf, paste(pathLoc, "_multiPlotDataOverview",".pdf", sep = ""))
    
    for (year in 2003:2017) {
      plotConcTimeDaily(eList, yearStart = year, yearEnd = year+1)  
      dev.print(pdf, paste(pathLoc, "_conc_", year, ".pdf", sep = ""))
      plotFluxTimeDaily(eList, yearStart = year, yearEnd = year+1)
      dev.print(pdf, paste(pathLoc, "_flux_", year, ".pdf", sep = ""))
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
    saveResults(paste(pathLoc, sep=""), eList)
    
    
    daily_name = paste(pathLoc, '_Daily.csv',sep="")
    sample_name = paste(pathLoc, '_Sample.csv',sep="")
    write.csv(eList[["Daily"]], daily_name)
    write.csv(eList[["Sample"]], sample_name)
# }
