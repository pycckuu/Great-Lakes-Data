graphics.off() 
par("mar") 

# library(doParallel)
# registerDoParallel(cores=6)

library(EGRET)

startDate <- "1996-01-01"
endDate <- "2018-01-01"


  river = "Portage_River"
  siteID = "04195500"
  parameter_cd = "00665"
  Sample <- readNWISSample(siteID,parameter_cd,startDate,endDate)
  #Gets earliest date from Sample record:
  #This is just one of many ways to assure the Daily record
  #spans the Sample record
  # Gather discharge data:
  Daily <- readNWISDaily(siteID,"00060",startDate,endDate)
  # Gather site and parameter information:
  
  # Here user must input some values for
  # the default (interactive=TRUE)
  INFO<- readNWISInfo(siteID,parameter_cd)
  # INFO$shortName <- "Portage River (US)"
  
  # Merge discharge with sample data:
  eList <- mergeReport(INFO, Daily, Sample)
  eList <- modelEstimation(eList)
    
    species = 'TP'
    pathLoc = paste("/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/Loadings US/Western/Portage R/EGRET/", river, "_", species, sep="")
    
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
    
    
    daily_name = paste(pathLoc, river, "_", short_name, '_Daily.csv',sep="")
    sample_name = paste(pathLoc, river, "_", short_name, '_Sample.csv',sep="")
    write.csv(eList[["Daily"]], daily_name)
    write.csv(eList[["Sample"]], sample_name)











