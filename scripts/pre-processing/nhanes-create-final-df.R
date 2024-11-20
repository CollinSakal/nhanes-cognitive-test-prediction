# Combines all feature data frames into one called df-final

# Libraries
library(tidyverse)

# Data - non-accelerometrty
df_noacc <- read_csv('data/nhanes/derived/df-non-accelerometry.csv')

# Data - wearable features from R code
# Get list of files
dir <- 'data/nhanes/derived/wearable-features-r'
fnames <- list.files(dir)
fpaths <- paste0(dir,'/',fnames)

df_features1 <- map_df(fpaths, read_csv)
df_features1 <- df_features1 %>% rename('SEQN' = 'id')

# Data - wearable features from Python code
# Get list of files
dir <- 'data/nhanes/derived/wearable-features-python'
fnames <- list.files(dir)
fpaths <- paste0(dir,'/',fnames)

df_features2 <- map_df(fpaths, read_csv)
df_features2 <- df_features2 %>% rename('SEQN' = 'id')

# Combine into one df
df_final <- 
  inner_join(df_noacc, df_features1, by = 'SEQN') %>% 
  inner_join(.,df_features2, by = 'SEQN')

# Save
write.csv(df_final, 'data/nhanes/derived/df-final-nhanes.csv', row.names = FALSE)

