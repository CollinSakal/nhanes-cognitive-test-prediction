# Splitting the accelerometry data into individual level files
# Combining the header files with the actual accelerometry files

# Libraries
library(haven)
library(tidyverse)

# Initialize stuff
days_5 <- 1440*5 # For the exclusion criteria, data are minute level

# Read in the non-accelerometry data to get the IDs needed
df_nonacc <- read_csv('data/nhanes/derived/df-non-accelerometry.csv')
ids_nonacc <- unique(df_nonacc$SEQN)

# Read in the 2011-12 accelerometry and header data 
# (only include IDs from the non accelerometry data to save RAM)
df11 <- read_xpt('data/nhanes/PAXMIN_G.XPT') %>%
  dplyr::filter(SEQN %in% ids_nonacc) %>% 
  dplyr::select(-PAXFLGSM, -PAXQFM) %>% 
  dplyr::mutate(PAXMTSM = ifelse(PAXMTSM == -0.01, 0, PAXMTSM))

df11_header <- read_xpt('data/nhanes/PAXHD_G.XPT') %>%
  dplyr::filter(SEQN %in% ids_nonacc) %>% 
  dplyr::select(-PAXSENID, -PAXSTS)

ids11_acc <- unique(df11$SEQN) # Accelerometry IDs
niter <- 1

# Loop over the IDs, join acc with header file, and save 
for(i in 1:length(ids11_acc)){
  
  # Join accelerometry with header df
  df11_temp <- left_join(df11 %>% filter(SEQN == ids11_acc[i]), 
                         df11_header %>% filter(SEQN == ids11_acc[i]), 
                         by = 'SEQN') 
  
  # Check exclusion criteria
  if(sum(df11_temp$PAXPREDM == 1 | df11_temp$PAXPREDM == 2) < days_5){
    
    print(paste0('Invalid data for ID ', ids11_acc[i]))
    
  }else{
    
    # Save the individual file 
    df_path <- paste0('data/nhanes/individual-accelerometer-files/', ids11_acc[i], '-accelerometry.csv')
    write.csv(df11_temp, df_path, row.names = FALSE) 
  }
  
  print(paste0(niter, ' of ', length(ids11_acc), ' files processed'))
  niter <- niter + 1
  
}

# Read in the 2013-2014 accelerometry and header data
# (only include IDs in the non accelerometry data to save RAM)
df13 <- read_xpt('data/nhanes/PAXMIN_H.XPT') %>%
  dplyr::filter(SEQN %in% ids_nonacc) %>% 
  dplyr::select(-PAXFLGSM, -PAXQFM) %>% 
  dplyr::mutate(PAXMTSM = ifelse(PAXMTSM == -0.01, 0, PAXMTSM))

df13_header <- read_xpt('data/nhanes/PAXHD_H.XPT') %>%
  dplyr::filter(SEQN %in% ids_nonacc) %>% 
  dplyr::select(-PAXSENID, -PAXSTS)

ids13_acc <- unique(df13$SEQN) # Accelerometry IDs
niter <- 1

# Loop over the IDs, join acc with header file, and save
for(i in 1:length(ids13_acc)){
  
  # Join accelerometry with header df
  df13_temp <- left_join(df13 %>% filter(SEQN == ids13_acc[i]), 
                         df13_header %>% filter(SEQN == ids13_acc[i]), 
                         by = 'SEQN') 
  
  # Check exclusion criteria
  if(sum(df13_temp$PAXPREDM == 1 | df13_temp$PAXPREDM == 2) < days_5){
    
    print(paste0('Invalid data for ID ', ids13_acc[i]))
    
  }else{
    
    # Save the individual file 
    df_path <- paste0('data/nhanes/individual-accelerometer-files/', ids13_acc[i], '-accelerometry.csv')
    write.csv(df13_temp, df_path, row.names = FALSE) 
    
  }
  
  print(paste0(niter, ' of ', length(ids13_acc), ' files processed'))
  niter <- niter + 1
  
}


