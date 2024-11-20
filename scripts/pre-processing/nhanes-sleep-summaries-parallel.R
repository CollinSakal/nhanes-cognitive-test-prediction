# Code for getting sleep summaries using parallel processing 

# Libraries
library(foreach)
library(tidyverse)
library(doParallel)

# Set seed
seed <- 19970507
set.seed(seed)

# Defining the number of cores to use (desktop has 24)
n_cores <- 18 

# Registering the cluster
registerDoParallel(n_cores)

# Set the directory and extract the list of accelerometry files
dir <- 'data/nhanes/individual-accelerometer-files'
flist <- paste0(dir,'/',list.files(dir))

# Run the parallel processing code
sleep_summaries_computation <- foreach(f = flist) %dopar% {
  
  # Set seed
  seed <- 19970507
  set.seed(seed)
  
  # Helper functions 
  source('scripts/utils/read-data1-nhanes.R')
  source('scripts/utils/get-sleep-summaries-nhanes.R')
  
  # Read in the data
  accdf <- read_data1(f) # Gets data into format for HMM input
  id_try <- accdf$id[1]  # Note: only one id per file
  
  # Try the modified sleep.summary2 function
  df_output <- try(
    sleep_summary(
      accdf, 
      interval = 60, 
      minhour = 16, 
      maxtry = 20, 
      sleepstart = 18, 
      sleepend = 10, 
      id_input = id_try
    )
  )
  
  # If there's an error then don't use data from that person
  #  otherwise save their data as a .csv file
  if('try-error' %in% class(df_output)){}else{
    if(nrow(df_output)<3){}else{
      outpath <- paste0('data/nhanes/derived/sleep-summaries/',id_try,'-sleep-summary.csv')
      write.csv(df_output, outpath, row.names = FALSE)
    }
  }
}

# Return to single core computations
stopImplicitCluster()
