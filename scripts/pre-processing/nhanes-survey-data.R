# Pre-processing the non-accelerometry data

# Libraries
library(haven)
library(tidyverse)

# Initializing stuff
age_cutoff <- 60
phq9_cutoff <- 10

# Data Imports (11/13 corresponds to the start year of the survey wave)
df_dems11 <- read_xpt('data/nhanes/DEMO_G.XPT')
df_dems13 <- read_xpt('data/nhanes/DEMO_H.XPT')
df_cogn11 <- read_xpt('data/nhanes/CFQ_G.XPT')
df_cogn13 <- read_xpt('data/nhanes/CFQ_H.XPT')
df_diab11 <- read_xpt('data/nhanes/DIQ_G.XPT')
df_diab13 <- read_xpt('data/nhanes/DIQ_H.XPT')
df_depr11 <- read_xpt('data/nhanes/DPQ_G.XPT')
df_depr13 <- read_xpt('data/nhanes/DPQ_H.XPT')
df_actv11 <- read_xpt('data/nhanes/PAQ_G.XPT')
df_actv13 <- read_xpt('data/nhanes/PAQ_H.XPT')
df_alch11 <- read_xpt('data/nhanes/ALQ_G.XPT')
df_alch13 <- read_xpt('data/nhanes/ALQ_H.XPT')
df_smkr11 <- read_xpt('data/nhanes/SMQ_G.XPT')
df_smkr13 <- read_xpt('data/nhanes/SMQ_H.XPT')
df_diag11 <- read_xpt('data/nhanes/MCQ_G.XPT')
df_diag13 <- read_xpt('data/nhanes/MCQ_H.XPT')

# Getting all the demographic variables we need
df_dems11 <- df_dems11 %>% 
  select(
    SEQN,                  # ID
    RIAGENDR,              # Sex
    RIDAGEYR,              # Age
    DMDEDUC2,              # Education
    DMDMARTL,              # Marital status
    INDHHIN2,              # Household income
  )

df_dems13 <- df_dems13 %>% 
  select(
    SEQN,                  # ID
    RIAGENDR,              # Sex
    RIDAGEYR,              # Age
    DMDEDUC2,              # Education
    DMDMARTL,              # Marital status
    INDHHIN2,              # Household income
  )

# Getting physical activity variables
df_actv11 <- df_actv11 %>% 
  select(
    SEQN,   # ID
    PAQ650, # Vigorous PA
  )

df_actv13 <- df_actv13 %>% 
  select(
    SEQN,   # ID
    PAQ650, # Vigorous PA
  )

# Getting all the smoking status variables we need
df_smkr11 <- df_smkr11 %>% 
  select(
    SEQN,   # ID
    SMQ020, # Smoked 100 cigarettes in life
    SMQ040, # Current smoker
  )

df_smkr13 <- df_smkr13 %>% 
  select(
    SEQN,   # ID
    SMQ020, # Smoked 100 cigarettes in life
    SMQ040, # Current smoker
  )

# Getting all the alcohol consumption variables we need
df_alch11 <- df_alch11 %>% 
  select(
    SEQN,    # ID
    ALQ110,  # At least 12 drinks in lifetime
    ALQ101,  # At least 12 drinks in the last year
    ALQ120Q, # Alcohol consumption over the last year
    ALQ120U, # Units for ^
  )

df_alch13 <- df_alch13 %>% 
  select(
    SEQN,    # ID
    ALQ110,  # At least 12 drinks in lifetime
    ALQ101,  # At least 12 drinks in the last year
    ALQ120Q, # Alcohol consumption over the last year
    ALQ120U, # Units for ^
  )

# Getting all the diabetes variables we need
df_diab11 <- df_diab11 %>% 
  select(
    SEQN,       # ID
    DIQ010,     # Ever been told by doctor has diabetes
  )

df_diab13 <- df_diab13 %>% 
  select(
    SEQN,       # ID
    DIQ010,     # Ever been told by doctor has diabetes
    
  )

# Getting all the depression variables we need
df_depr11 <- df_depr11 %>% 
  select(
    SEQN,       # ID
    DPQ010,     # PHQ1
    DPQ020,     # PHQ2
    DPQ030,     # PHQ3
    DPQ040,     # PHQ4
    DPQ050,     # PHQ5
    DPQ060,     # PHQ6
    DPQ070,     # PHQ7
    DPQ080,     # PHQ8
    DPQ090,     # PHQ9
  )

df_depr13 <- df_depr13 %>% 
  select(
    SEQN,       # ID
    DPQ010,     # PHQ1
    DPQ020,     # PHQ2
    DPQ030,     # PHQ3
    DPQ040,     # PHQ4
    DPQ050,     # PHQ5
    DPQ060,     # PHQ6
    DPQ070,     # PHQ7
    DPQ080,     # PHQ8
    DPQ090,     # PHQ9
  )

# Getting all the cognition variables we need
df_cogn11 <- df_cogn11 %>% 
  select(
    SEQN,        # ID
    CFDCST1,     # CERAD: Score trial 1
    CFDCST2,     # CERAD: Score trial 2
    CFDCST3,     # CERAD: Score trial 3
    CFDCSR,      # CERAD: Score delayed recall
    CFDAST,      # Animal fluency score
    CFDDS,       # Digit symbol score
  )

df_cogn13 <- df_cogn13 %>% 
  select(
    SEQN,        # ID
    CFDCST1,     # CERAD: Score trial 1
    CFDCST2,     # CERAD: Score trial 2
    CFDCST3,     # CERAD: Score trial 3
    CFDCSR,      # CERAD: Score delayed recall
    CFDAST,      # Animal fluency score
    CFDDS,       # Digit symbol score
  )

# Cleaning the demographic variables
# Binarize sex variable: 0 = male, 1 = female
df_dems11$RIAGENDR <- df_dems11$RIAGENDR - 1
df_dems13$RIAGENDR <- df_dems13$RIAGENDR - 1

# Education: set "Refused" (7) and "I don't know" (9) to missing
df_dems11$DMDEDUC2 <- ifelse(df_dems11$DMDEDUC2 %in% c(7,9), NA, df_dems11$DMDEDUC2)
df_dems13$DMDEDUC2 <- ifelse(df_dems13$DMDEDUC2 %in% c(7,9), NA, df_dems13$DMDEDUC2)

# Marital status: set "Refused" (77) and "I don't know" (99) to missing
df_dems11$DMDMARTL <- ifelse(df_dems11$DMDMARTL %in% c(77,99), NA, df_dems11$DMDMARTL)
df_dems13$DMDMARTL <- ifelse(df_dems13$DMDMARTL %in% c(77,99), NA, df_dems13$DMDMARTL)

# Marital status:
#    1 = married or living with partner
#    2 = divorced or seperated
#    3 = widowed
#    4 = never married
df_dems11 <- df_dems11 %>% 
  mutate(
    DMDMARTL = case_when(
      DMDMARTL %in% c(1,6) ~ 1,
      DMDMARTL %in% c(3,4) ~ 2,
      DMDMARTL == 2 ~ 3,
      DMDMARTL == 5 ~ 4,
      )
  )

df_dems13 <- df_dems13 %>% 
  mutate(
    DMDMARTL = case_when(
      DMDMARTL %in% c(1,6) ~ 1,
      DMDMARTL %in% c(3,4) ~ 2,
      DMDMARTL == 2 ~ 3,
      DMDMARTL == 5 ~ 4,
    )
  )

# Household income: set "Refused" (77) and "I don't know (99) to missing
#  Set imprecise income values <>20k (12, 13) to missing as well
df_dems11$INDHHIN2 <- ifelse(df_dems11$INDHHIN2 %in% c(12,13,77,99), NA, df_dems11$INDHHIN2)
df_dems13$INDHHIN2 <- ifelse(df_dems13$INDHHIN2 %in% c(12,13,77,99), NA, df_dems13$INDHHIN2)

# Shift the remaining income values down so they are orindally spaced by 1
df_dems11$INDHHIN2 <- ifelse(df_dems11$INDHHIN2 == 14, 11, df_dems11$INDHHIN2)
df_dems11$INDHHIN2 <- ifelse(df_dems11$INDHHIN2 == 15, 12, df_dems11$INDHHIN2)
df_dems13$INDHHIN2 <- ifelse(df_dems13$INDHHIN2 == 14, 11, df_dems13$INDHHIN2)
df_dems13$INDHHIN2 <- ifelse(df_dems13$INDHHIN2 == 15, 12, df_dems13$INDHHIN2)

# Categorize as 0-45k, 45k-100k, 100k+
df_dems11 <- df_dems11 %>% 
  mutate(
    INDHHIN2 = case_when(
      INDHHIN2 %in% seq(1,7,1) ~ 0,
      INDHHIN2 %in% seq(8,11,1) ~ 1,
      INDHHIN2 == 12 ~ 2
    )
  )

df_dems13 <- df_dems13 %>% 
  mutate(
    INDHHIN2 = case_when(
      INDHHIN2 %in% seq(1,7,1) ~ 0,
      INDHHIN2 %in% seq(8,11,1) ~ 1,
      INDHHIN2 == 12 ~ 2
    )
  )

# Cleaning all depression (PHQ9) variables
# For all PHQ9 variables: set "Refused" (7) and "I don't know" (9) to missing
df_depr11 <- df_depr11 %>% 
  mutate(
    DPQ010 = ifelse(DPQ010 %in% c(7,9),NA, DPQ010),
    DPQ020 = ifelse(DPQ020 %in% c(7,9),NA, DPQ020),
    DPQ030 = ifelse(DPQ030 %in% c(7,9),NA, DPQ030),
    DPQ040 = ifelse(DPQ040 %in% c(7,9),NA, DPQ040),
    DPQ050 = ifelse(DPQ050 %in% c(7,9),NA, DPQ050),
    DPQ060 = ifelse(DPQ060 %in% c(7,9),NA, DPQ060),
    DPQ070 = ifelse(DPQ070 %in% c(7,9),NA, DPQ070),
    DPQ080 = ifelse(DPQ080 %in% c(7,9),NA, DPQ080),
    DPQ090 = ifelse(DPQ090 %in% c(7,9),NA, DPQ090),
    phq = DPQ010+DPQ020+DPQ030+DPQ040+DPQ050+DPQ060+DPQ080+DPQ090
  )

table(df_depr11$phq)

df_depr13 <- df_depr13 %>% 
  mutate(
    DPQ010 = ifelse(DPQ010 %in% c(7,9),NA, DPQ010),
    DPQ020 = ifelse(DPQ020 %in% c(7,9),NA, DPQ020),
    DPQ030 = ifelse(DPQ030 %in% c(7,9),NA, DPQ030),
    DPQ040 = ifelse(DPQ040 %in% c(7,9),NA, DPQ040),
    DPQ050 = ifelse(DPQ050 %in% c(7,9),NA, DPQ050),
    DPQ060 = ifelse(DPQ060 %in% c(7,9),NA, DPQ060),
    DPQ070 = ifelse(DPQ070 %in% c(7,9),NA, DPQ070),
    DPQ080 = ifelse(DPQ080 %in% c(7,9),NA, DPQ080),
    DPQ090 = ifelse(DPQ090 %in% c(7,9),NA, DPQ090),
    phq = DPQ010+DPQ020+DPQ030+DPQ040+DPQ050+DPQ060+DPQ080+DPQ090
  )

table(df_depr13$phq)

# Creating the final PHQ9 score, binarizing into a depression variable
df_depr11 <- df_depr11 %>% 
  mutate(
    depressed = if_else(phq >= phq9_cutoff, 1, 0)
  ) %>% 
  select(
    SEQN, depressed
  )

df_depr13 <- df_depr13 %>% 
  mutate(
    depressed = if_else(phq >= phq9_cutoff, 1, 0)
  ) %>% 
  select(
    SEQN, depressed
  )

# Cleaning physical activity variables and making them binary
df_actv11$PAQ650 <- ifelse(df_actv11$PAQ650 %in% c(1,2), df_actv11$PAQ650, NA)
df_actv13$PAQ650 <- ifelse(df_actv13$PAQ650 %in% c(1,2), df_actv13$PAQ650, NA)

df_actv11$PAQ650 <- ifelse(df_actv11$PAQ650 == 2, 0, df_actv11$PAQ650)
df_actv13$PAQ650 <- ifelse(df_actv13$PAQ650 == 2, 0, df_actv13$PAQ650)

# Cleaning the diabetes variables
# For diabetic status set 'I don't know' (9) and 'Refused' (7) to missing
#  Also binarize s.t. 0 = no and 1 = yes
#  Also set "Borderline" (3) to no (0)
df_diab11$DIQ010 <- ifelse(df_diab11$DIQ010 %in% c(7,9), NA, df_diab11$DIQ010)
df_diab13$DIQ010 <- ifelse(df_diab13$DIQ010 %in% c(7,9), NA, df_diab13$DIQ010)
df_diab11$DIQ010 <- ifelse(df_diab11$DIQ010 %in% c(2,3), 0, df_diab11$DIQ010)
df_diab13$DIQ010 <- ifelse(df_diab13$DIQ010 %in% c(2,3), 0, df_diab13$DIQ010)

# Cleaning the smoking status variables
# For smoking status set 'I don't know' (9) and 'Refused' (7) to missing
df_smkr11$SMQ020 <- ifelse(df_smkr11$SMQ020 %in% c(7,9), NA, df_smkr11$SMQ020)
df_smkr13$SMQ020 <- ifelse(df_smkr13$SMQ020 %in% c(7,9), NA, df_smkr13$SMQ020)
df_smkr11$SMQ040 <- ifelse(df_smkr11$SMQ040 %in% c(7,9), NA, df_smkr11$SMQ040)
df_smkr13$SMQ040 <- ifelse(df_smkr13$SMQ040 %in% c(7,9), NA, df_smkr13$SMQ040)

table(df_smkr11$SMQ020)
table(df_smkr13$SMQ040)
table(df_smkr11$SMQ020)
table(df_smkr13$SMQ040)

# Smoking status (1 = smokes, 0 = does not)
df_smkr11 <- df_smkr11 %>% 
  mutate(
    smoker = case_when(
      SMQ020 == 2 ~ 0,
      SMQ020 == 1 & SMQ040 == 3 ~ 0,
      SMQ040 %in% c(1,2) ~ 1,
      .default = NA
    )
  )

df_smkr13 <- df_smkr13 %>% 
  mutate(
    smoker = case_when(
      SMQ020 == 2 ~ 0,
      SMQ020 == 1 & SMQ040 == 3 ~ 0,
      SMQ040 %in% c(1,2) ~ 1,
      .default = NA
    )
  )

# Cleaning alcohol consumption variables
# For consumption freq/units set 'I don't know' (9) and 'Refused' (7) to missing
df_alch11$ALQ120Q <- ifelse(df_alch11$ALQ120Q %in% c(777, 999), NA, df_alch11$ALQ120Q)
df_alch13$ALQ120Q <- ifelse(df_alch13$ALQ120Q %in% c(777, 999), NA, df_alch13$ALQ120Q)
df_alch11$ALQ120U <- ifelse(df_alch11$ALQ120U %in% c(777, 999), NA, df_alch11$ALQ120U)
df_alch13$ALQ120U <- ifelse(df_alch13$ALQ120U %in% c(777, 999), NA, df_alch13$ALQ120U)

# Alcohol consumption (1 = drinks, 0 = does not)
df_alch11 <- df_alch11 %>% 
  mutate(
    drinker = case_when(
      ALQ101 == 2 ~ 0,
      ALQ101 == 1 & ALQ120Q == 0 ~ 0,
      ALQ101 == 1 & ALQ120Q != 0 ~ 1,
      .default = NA
    )
  )

df_alch13 <- df_alch13 %>% 
  mutate(
    drinker = case_when(
      ALQ101 == 2 ~ 0,
      ALQ101 == 1 & ALQ120Q == 0 ~ 0,
      ALQ101 == 1 & ALQ120Q != 0 ~ 1,
      .default = NA
    )
  )

# Cleaning the medical conditions variables
# Setting all 'I don't know' (9) and 'Refused' (7) to missing
df_diag11$MCQ160A <- ifelse(df_diag11$MCQ160A %in% c(7,9), NA, df_diag11$MCQ160A)
df_diag11$MCQ160C <- ifelse(df_diag11$MCQ160C %in% c(7,9), NA, df_diag11$MCQ160C)
df_diag11$MCQ160E <- ifelse(df_diag11$MCQ160E %in% c(7,9), NA, df_diag11$MCQ160E)
df_diag11$MCQ160F <- ifelse(df_diag11$MCQ160F %in% c(7,9), NA, df_diag11$MCQ160F)

df_diag13$MCQ160A <- ifelse(df_diag13$MCQ160A %in% c(7,9), NA, df_diag13$MCQ160A)
df_diag13$MCQ160C <- ifelse(df_diag13$MCQ160C %in% c(7,9), NA, df_diag13$MCQ160C)
df_diag13$MCQ160E <- ifelse(df_diag13$MCQ160E %in% c(7,9), NA, df_diag13$MCQ160E)
df_diag13$MCQ160F <- ifelse(df_diag13$MCQ160F %in% c(7,9), NA, df_diag13$MCQ160F)

# Binarizing so 1 = has condition, 0 = does not
df_diag11$MCQ160A <- ifelse(df_diag11$MCQ160A == 2, 0, df_diag11$MCQ160A)
df_diag11$MCQ160C <- ifelse(df_diag11$MCQ160C == 2, 0, df_diag11$MCQ160C)
df_diag11$MCQ160E <- ifelse(df_diag11$MCQ160E == 2, 0, df_diag11$MCQ160E)
df_diag11$MCQ160F <- ifelse(df_diag11$MCQ160F == 2, 0, df_diag11$MCQ160F)

df_diag13$MCQ160A <- ifelse(df_diag13$MCQ160A == 2, 0, df_diag13$MCQ160A)
df_diag13$MCQ160C <- ifelse(df_diag13$MCQ160C == 2, 0, df_diag13$MCQ160C)
df_diag13$MCQ160E <- ifelse(df_diag13$MCQ160E == 2, 0, df_diag13$MCQ160E)
df_diag13$MCQ160F <- ifelse(df_diag13$MCQ160F == 2, 0, df_diag13$MCQ160F)

# Creating cognition scores
df_cogn11$cerad_score <- 
  df_cogn11$CFDCST1+
  df_cogn11$CFDCST2+
  df_cogn11$CFDCST3+
  df_cogn11$CFDCSR

df_cogn13$cerad_score <- 
  df_cogn13$CFDCST1+
  df_cogn13$CFDCST2+
  df_cogn13$CFDCST3+
  df_cogn13$CFDCSR

df_cogn11$af_score <- df_cogn11$CFDAST
df_cogn13$af_score <- df_cogn13$CFDAST

df_cogn11$dsst_score <- df_cogn11$CFDDS
df_cogn13$dsst_score <- df_cogn13$CFDDS

df_cogn11 <- df_cogn11 %>% select(SEQN, cerad_score, af_score, dsst_score)
df_cogn13 <- df_cogn13 %>%  select(SEQN, cerad_score, af_score, dsst_score)


# Combining the 2011 and 2013 data frames
df11 <- 
  left_join(df_cogn11, df_dems11, by = 'SEQN')%>% 
  left_join(.,df_actv11, by = 'SEQN') %>% 
  left_join(.,df_diab11, by = 'SEQN') %>% 
  left_join(.,df_depr11, by = 'SEQN') %>% 
  left_join(.,df_smkr11, by = 'SEQN') %>% 
  left_join(.,df_alch11, by = 'SEQN') %>% 
  left_join(.,df_diag11, by = 'SEQN')
df13 <- 
  left_join(df_cogn13, df_dems13, by = 'SEQN') %>% 
  left_join(.,df_actv13, by = 'SEQN') %>% 
  left_join(.,df_diab13, by = 'SEQN') %>% 
  left_join(.,df_depr13, by = 'SEQN') %>% 
  left_join(.,df_smkr13, by = 'SEQN') %>% 
  left_join(.,df_alch13, by = 'SEQN') %>% 
  left_join(.,df_diag13, by = 'SEQN')

df_final <- bind_rows(df11, df13)

# Get cognitive score cutoffs
cerad_cut <- quantile(df_final$cerad_score, .25, na.rm=TRUE)
af_cut <- quantile(df_final$af_score, .25, na.rm=TRUE)
dsst_cut <- quantile(df_final$dsst_score, .25, na.rm=TRUE)

print(paste0('Recall cutoff: ', cerad_cut))
print(paste0('AFT cutoff: ', af_cut))
print(paste0('DSST cutoff: ', dsst_cut))

# Creating 'low' variables indicating poor cognitive performance based on the 25th quantile
df_final$cerad_low <- ifelse(df_final$cerad_score <= cerad_cut,1,0)
df_final$af_low <- ifelse(df_final$af_score <= af_cut,1,0)
df_final$dsst_low <- ifelse(df_final$dsst_score <= dsst_cut,1,0)

# Exclusion criteria, renaming
df_final <- df_final %>% 
  filter(RIDAGEYR >= age_cutoff) %>% 
  drop_na(cerad_score, dsst_score, af_score) %>% 
  rename(
    age = RIDAGEYR,
    sex = RIAGENDR,
    education = DMDEDUC2,
    household_income = INDHHIN2,
    marital_status = DMDMARTL,
    diabetes = DIQ010,
    arthritis = MCQ160A,
    heart_disease = MCQ160C,
    heart_attack = MCQ160E,
    stroke  = MCQ160F,
    vigorous_pa = PAQ650,
  )  %>% 
  mutate(
    education = case_when(
      education %in% c(1,2) ~ 1,
      education == 3 ~ 2,
      education == 4 ~ 3,
      education == 5 ~ 4,
      .default = NA
    )
  ) %>% 
  select(
    SEQN,
    dsst_low,
    cerad_low,
    af_low,
    age, 
    sex,
    education, 
    household_income,
    marital_status,
    depressed,
    diabetes,
    arthritis,
    heart_disease,
    heart_attack,
    stroke,
    smoker,
    drinker,
  )

# Brief checks
nrow(df_final)                  
colMeans(is.na(df_final))*100  

# Save
write.csv(df_final, 'data/nhanes/derived/df-non-accelerometry.csv', row.names = FALSE)

