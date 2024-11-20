# File for reading data prior to using the HMM

# Libraries
library(tidyverse)
library(lubridate)

# Function
read_data1 <- function(acc_filepath){
  
  # Get the file and necessary columns
  df <- read_csv(acc_filepath, show_col_types = FALSE) %>% 
    dplyr::select(
      SEQN, 
      PAXMTSM, 
      PAXFTIME, 
      PAXFDAY
    ) %>% 
    as.data.frame()
  
  # Note: the real start years/months aren't provided, 
  #       chose 2017 because it started on a Sunday 
  #       which corresponds to the weekday numbering 
  #       in the NHANES data (Sunday = 1, Monday = 2, ..)
  
  start_ymd <- paste0("2017-01-", df$PAXFDAY[1])
  start_hms <- df$PAXFTIME[1]
  
  start_date_ymdhms <- as.POSIXlt(paste(start_ymd, start_hms), 
                                  format = "%Y-%m-%d %H:%M:%OS",
                                  tz = "UTC") 
  
  # Add date column (note 'by' must match acc-interval)
  df$date_ymdhms <- seq(start_date_ymdhms, by='min', length=nrow(df))
  
  # Get correct names and formats so it works with the sleep/wake alg
  df <- df %>% 
    dplyr::select(SEQN, date_ymdhms, PAXMTSM) %>% 
    dplyr::rename(
      'id' = 'SEQN',
      'time' = 'date_ymdhms',
      'count' = 'PAXMTSM',
    )
  
  df$time <- as.POSIXlt(df$time, tz="UTC")
  
  # Output the data frame
  return(df)
}