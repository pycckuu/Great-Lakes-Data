graphics.off() 
par("mar") 


library(EGRET)

species_list <- list()
species_list[['Cl']]<- 'Cl'
species_list[['SRP']]<- 'Soluble Reactive Phosphorus'
species_list[['TP']]<- 'Total Phosphorus'

for (species in names(species_list)){
  for (river in c("Cuyahoga_River", "Maumee_River", "Portage_River", "Raisin_River", "Sandusky_River", "Vermilion_River")){
  # for (river in c("Portage_River")){
    flow_name = paste("proc_data/Flow_", river, "_data.csv", sep = "")
    sample_name = paste("proc_data/", species, "_", river, "_data.csv", sep = "")
    
    Daily <- readUserDaily('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', flow_name, qUnit = 1)
    Sample <- readUserSample('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', sample_name)
    Sample <- removeDuplicates(Sample)
    
    INFO<-data.frame(0)
    INFO$shortName<- gsub("_", " ", river)
    INFO$param.units<-'mg/l'
    INFO$paramShortName<- species_list$species
    INFO$staAbbrev<- '0'
    INFO$constitAbbrev<- species
    INFO$drainSqKm <- 0
    
    eList <- mergeReport(INFO, Daily,Sample)
    eList <- setPA(eList, paStart = 1, paLong = 12)
    eList <- modelEstimation(eList)
    
    multiPlotDataOverview(eList)
    dev.print(pdf, paste("plots/", river,"/", species , "/", "multiPlotDataOverview",".pdf", sep = ""))
    
    
    
    for (year in 2003:2016) {
      plotConcTimeDaily(eList, yearStart = year, yearEnd = year+1)  
      dev.print(pdf, paste("plots/", river,"/", species, "/", year, "_conc.pdf", sep = ""))
      plotFluxTimeDaily(eList, yearStart = year, yearEnd = year+1)
      dev.print(pdf, paste("plots/", river,"/", species, "/", year, "_flux.pdf", sep = ""))
    }
    
    plotConcPred(eList)
    dev.print(pdf, paste("plots/", river,"/", species, "/plotConcPred.pdf", sep = ""))
    plotFluxPred(eList)
    dev.print(pdf, paste("plots/", river,"/", species, "/plotFluxPred.pdf", sep = ""))
    plotResidPred(eList)
    dev.print(pdf, paste("plots/", river,"/", species, "/plotResidPred.pdf", sep = ""))
    plotResidQ(eList)
    dev.print(pdf, paste("plots/", river,"/", species, "/plotResidQ.pdf", sep = ""))
    plotResidTime(eList)
    dev.print(pdf, paste("plots/", river,"/", species, "/plotResidTime.pdf", sep = ""))
    boxResidMonth(eList)
    dev.print(pdf, paste("plots/", river,"/", species, "/boxResidMonth.pdf", sep = ""))
    boxConcThree(eList)
    dev.print(pdf, paste("plots/", river,"/", species, "/boxConcThree.pdf", sep = ""))
    
    plotConcHist(eList)
    dev.print(pdf, paste("plots/", river,"/", species, "/plotConcHist",".pdf", sep = ""))
    plotFluxHist(eList)
    dev.print(pdf, paste("plots/", river,"/", species, "/plotFluxHist",".pdf", sep = ""))
    saveResults("plots/", river,"/", species, "/", eList)
  }
}











