install.packages("lfstat")
install.packages("RcmdrPlugin.lfstat")
require(lfstat)
require(RcmdrPlugin.lfstat)

setwd("/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/Base Flow/data")
tbl <-list.files(recursive = T, pattern = ".csv$")
for(i in tbl){
  print(i)
  Dataset <- 
    read.table(i,
               header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
  LFdata <- createlfobj(x =ts(Dataset$Flow..CFS) , startdate = "01/01/1996",
                        hyearstart =1,baseflow =TRUE)
  write.csv(LFdata, file = paste("Base", i, sep=""))
  }


# rm(list=ls())
# 
# Dataset <- 
#   read.table("/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/Base Flow/data/Flow_Grand_R_CFS.csv",
#              header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
# LFdata <- createlfobj(x =ts(Dataset$Flow..CFS) , startdate = "01/01/1996",
#                       hyearstart =1,baseflow =TRUE)
# 
# write.csv(LFdata, file = "/Users/imarkelo/git/Great-Lakes-Data/post_proc_scripts/Base Flow/data/BaseFlow_Grand_R_CFS.csv")
