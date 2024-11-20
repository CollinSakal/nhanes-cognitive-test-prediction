# Calculating sleep metrics using parallel processing

# Libraries
library(foreach)
library(tidyverse)
library(doParallel)

# Defining the number of cores to use (desktop has 24)
n_cores <- 18 

# Registering the cluster
registerDoParallel(n_cores)

# Note: the sleep IDs are a subset of the accelerometry IDs 
ids_slp <- substr(list.files('data/nhanes/derived/sleep-summaries'),1,5)

# Run the parallel processing code
sleep_covariates_computation <- foreach(id = ids_slp) %dopar% {
  
  # Helper functions
  source('scripts/utils/read-data2-nhanes.R')
  source('scripts/utils/get-wearable-features-nhanes.R')
  
  # Define directory
  dir_slp <- 'data/nhanes/derived/sleep-summaries/'
  dir_acc <- 'data/nhanes/individual-accelerometer-files/'
  
  # Get sleep path for one person
  slp_path <- paste0(dir_slp,id,'-sleep-summary.csv')
  acc_path <- paste0(dir_acc,id,'-accelerometry.csv')
  
  # Get data
  df_slp <- read_csv(slp_path, show_col_types=FALSE)
  df_acc <- read_data2(acc_path)

  df_output <- try(get_wearable_features(df_slp=df_slp, df_acc=df_acc, id=id))
  
  if('try-error' %in% class(df_output)){}else{
    outpath <- paste0('data/nhanes/derived/wearable-features-r/',id,'-wearable-features.csv')
    write.csv(df_output, outpath, row.names = FALSE)
  }

}

# Return to single core computations
stopImplicitCluster()
