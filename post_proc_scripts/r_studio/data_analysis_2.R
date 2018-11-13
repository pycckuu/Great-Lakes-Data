graphics.off() 
par("mar") 


library(EGRET)


for (river in c("Cuyahoga_River", "Maumee_River", "Portage_River", "Raisin_River", "Sandusky_River", "Vermilion_River")){
  flow_name = paste("Flow_", river, "_data.csv", sep = "")
  sample_name = paste("TP_", river, "_data.csv", sep = "")
  
  Daily <- readUserDaily('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', flow_name, qUnit = 1)
  Sample <- readUserSample('/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/r_studio/', sample_name)
  Sample <- removeDuplicates(Sample)
  
  INFO<-data.frame(0)
  INFO$shortName<- gsub("_", " ", river)
  INFO$param.units<-'mg/l'
  INFO$paramShortName<- 'Total Phosphorus'
  INFO$staAbbrev<- '0'
  INFO$constitAbbrev<- 'TP'
  INFO$drainSqKm <- 0
  
  eList <- mergeReport(INFO, Daily,Sample)
  eList <- setPA(eList, paStart = 1, paLong = 12)
  eList <- modelEstimation(eList)
  
  multiPlotDataOverview(eList)
  dev.print(pdf, paste("plots/", river,"/TP_multiPlotDataOverview",".pdf", sep = ""))
  
  
  
  for (year in 2003:2016) {
    plotConcTimeDaily(eList, yearStart = year, yearEnd = year+1)  
    dev.print(pdf, paste("plots/", river,"/", year, "_TP_conc.pdf", sep = ""))
    plotFluxTimeDaily(eList, yearStart = year, yearEnd = year+1)
    dev.print(pdf, paste("plots/", river,"/", year, "_TP_flux.pdf", sep = ""))
  }
  
  plotConcPred(eList)
  dev.print(pdf, paste("plots/", river,"/TP","_plotConcPred.pdf", sep = ""))
  plotFluxPred(eList)
  dev.print(pdf, paste("plots/", river,"/TP","_plotFluxPred.pdf", sep = ""))
  plotResidPred(eList)
  dev.print(pdf, paste("plots/", river,"/TP","_plotResidPred.pdf", sep = ""))
  plotResidQ(eList)
  dev.print(pdf, paste("plots/", river,"/TP","_plotResidQ.pdf", sep = ""))
  plotResidTime(eList)
  dev.print(pdf, paste("plots/", river,"/TP","_plotResidTime.pdf", sep = ""))
  boxResidMonth(eList)
  dev.print(pdf, paste("plots/", river,"/TP","_boxResidMonth.pdf", sep = ""))
  boxConcThree(eList)
  dev.print(pdf, paste("plots/", river,"/TP","_boxConcThree.pdf", sep = ""))
  
  plotConcHist(eList)
  dev.print(pdf, paste("plots/", river,"/TP","_plotConcHist",".pdf", sep = ""))
  plotFluxHist(eList)
  dev.print(pdf, paste("plots/", river,"/TP","_plotFluxHist",".pdf", sep = ""))
  
}






