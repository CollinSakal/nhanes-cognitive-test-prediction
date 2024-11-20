# File for reading in data prior to getting the features

# Libraries
library(tidyverse)
library(lubridate)

# Function
read_data2 <- function(acc_filepath){
  
  # Outputs additional columns comapred to read_data
  
  # Get the file and necessary columns
  df <- read_csv(acc_filepath, show_col_types = FALSE) %>% 
    dplyr::select(
      SEQN,           # ID
      PAXMTSM,        # MIMS
      PAXFTIME,       # Start time
      PAXFDAY,        # Start date
      PAXPREDM,       # Wear/nonwear prediction
      PAXLXMM         # Ambient light
    ) %>% 
    as.data.frame()
  
  # Note: the real start dates aren't provided, I chose 2017 
  #       because the year started on a Sunday which corresponds 
  #       to the weekday numbering in NHANES (Sunday = 1, Monday = 2, ...)
  start_ymd <- paste0("2017-01-", df$PAXFDAY[1]) 
  start_hms <- df$PAXFTIME[1]                    
  
  start_date_ymdhms <- as.POSIXlt(paste(start_ymd, start_hms), 
                                  format = "%Y-%m-%d %H:%M:%OS",
                                  tz = 'UTC') 
  
  # Populate date column between start date and remaining observations
  df$date_ymdhms <- seq(start_date_ymdhms, by='min', length=nrow(df))
  
  # Select necessary cols, rename, and output the df
  df <- df %>% 
    dplyr::select(SEQN, date_ymdhms, PAXMTSM, PAXPREDM, PAXLXMM) %>% 
    dplyr::rename(
      'id' = 'SEQN',
      'time' = 'date_ymdhms',
      'count' = 'PAXMTSM',
      'wear' = 'PAXPREDM',
      'lux' = 'PAXLXMM'
    )
  
  df$time <- as.POSIXlt(df$time, tz='UTC')
  
  return(df)
}