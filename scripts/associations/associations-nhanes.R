# File for looking at associations in the NHANES 

# Libraries
library(tidyverse)
library(gtable)
library(grid)
library(gridExtra)
library(ggpubr)
library(broom)
library(patchwork)

# Data
df <- read_csv('data/nhanes/derived/df-final-nhanes.csv') %>% drop_na()

covariate_names <- 
  colnames(
    df %>% 
      select(!starts_with('frq')) %>% 
      select(!ends_with('score')) %>% 
      select(!ends_with('low')) %>%
      select(!c(
        'age','sex','education','marital_status', 'household_income',
        'drinker', 'smoker', 'arthritis', 'heart_attack', 
        'stroke', 'diabetes', 'depressed', 'heart_disease'
        
      )) %>% 
      select(!'SEQN')
    )

df <- df %>% 
  mutate(
    education=factor(education),
    marital_status=factor(marital_status),
    household_income=factor(household_income),
  ) %>% 
  mutate(across(all_of(covariate_names), ~scale(.) %>% as.vector))

# Modelling ----
df_results <- tibble(
  target=character(), 
  model=character(),
  term=character(), 
  estimate=numeric(), 
  std.error=numeric(),
  p.value=numeric(),
)

for(cov in covariate_names){
  
  # ****************************************************************************
  # UNIVARIABLE MODELS
  # ****************************************************************************
  
  # DSST
  formula <- as.formula(paste('dsst_low ~', cov))
  model <- glm(formula, data=df, family=binomial)
  tidy_model <- tidy(model)
  
  covariate_result <- tidy_model %>%
    filter(term == cov) %>%
    select(term, estimate, std.error, p.value) %>%
    mutate(target='DSST', model='Univariable') 

  df_results <- bind_rows(df_results, covariate_result)
  
  # CERAD
  formula <- as.formula(paste('cerad_low ~', cov))
  model <- glm(formula, data=df, family=binomial)
  tidy_model <- tidy(model)
  
  covariate_result <- tidy_model %>%
    filter(term == cov) %>%
    select(term, estimate, std.error, p.value) %>%
    mutate(target='CERAD', model='Univariable') 
  
  df_results <- bind_rows(df_results, covariate_result)
  
  # AFT
  formula <- as.formula(paste('af_low ~', cov))
  model <- glm(formula, data=df, family=binomial)
  tidy_model <- tidy(model)
  
  covariate_result <- tidy_model %>%
    filter(term == cov) %>%
    select(term, estimate, std.error, p.value) %>%
    mutate(target='AFT', model='Univariable') 
  
  df_results <- bind_rows(df_results, covariate_result)
  
  # ****************************************************************************
  # DEMOGRAPHIC MODELS
  # ****************************************************************************
  
  # DSST
  formula <- as.formula(
    paste('dsst_low ~ age+sex+education+marital_status+household_income+', cov)
  )
  model <- glm(formula, data=df, family=binomial)
  tidy_model <- tidy(model)
  
  covariate_result <- tidy_model %>%
    filter(term == cov) %>%
    select(term, estimate, std.error, p.value) %>%
    mutate(target='DSST', model='Demographics') 
  
  df_results <- bind_rows(df_results, covariate_result)
  
  # CERAD
  formula <- as.formula(
    paste('cerad_low ~ age+sex+education+marital_status+household_income+', cov)
  )
  model <- glm(formula, data=df, family=binomial)
  tidy_model <- tidy(model)
  
  covariate_result <- tidy_model %>%
    filter(term == cov) %>%
    select(term, estimate, std.error, p.value) %>%
    mutate(target='CERAD', model='Demographics') 
  
  df_results <- bind_rows(df_results, covariate_result)
  
  # AFT
  formula <- as.formula(
    paste('af_low ~ age+sex+education+marital_status+household_income+', cov)
  )
  model <- glm(formula, data=df, family=binomial)
  tidy_model <- tidy(model)
  
  covariate_result <- tidy_model %>%
    filter(term == cov) %>%
    select(term, estimate, std.error, p.value) %>%
    mutate(target='AFT', model='Demographics') 
  
  df_results <- bind_rows(df_results, covariate_result)
  
  # ****************************************************************************
  # FULL MODELS
  # ****************************************************************************
  
  # DSST
  formula <- as.formula(
    paste('dsst_low ~ age+sex+education+marital_status+household_income+diabetes+depressed+heart_disease+arthritis+smoker+drinker+', cov)
  )
  model <- glm(formula, data=df, family=binomial)
  tidy_model <- tidy(model)
  
  covariate_result <- tidy_model %>%
    filter(term == cov) %>%
    select(term, estimate, std.error, p.value) %>%
    mutate(target='DSST', model='Full') 
  
  df_results <- bind_rows(df_results, covariate_result)
  
  # CERAD
  formula <- as.formula(
    paste('cerad_low ~ age+sex+education+marital_status+household_income+diabetes+depressed+heart_disease+arthritis+smoker+drinker+', cov)
  )
  model <- glm(formula, data=df, family=binomial)
  tidy_model <- tidy(model)
  
  covariate_result <- tidy_model %>%
    filter(term == cov) %>%
    select(term, estimate, std.error, p.value) %>%
    mutate(target='CERAD', model='Full') 
  
  df_results <- bind_rows(df_results, covariate_result)
  
  # AFT
  formula <- as.formula(
    paste('af_low ~ age+sex+education+marital_status+household_income+diabetes+depressed+heart_disease+arthritis+smoker+drinker+', cov)
  )
  model <- glm(formula, data=df, family=binomial)
  tidy_model <- tidy(model)
  
  covariate_result <- tidy_model %>%
    filter(term == cov) %>%
    select(term, estimate, std.error, p.value) %>%
    mutate(target='AFT', model='Full') 
  
  df_results <- bind_rows(df_results, covariate_result)
  
}

# Adding ORs, CIs, and significance stars. ADjust p-values
df_results <- df_results %>% 
  group_by(target,model) %>% 
  mutate(p.value=p.adjust(p.value, method='BH')) %>% 
  ungroup() %>% 
  mutate(
    odds.ratio=exp(estimate),
    ci.upper=exp(estimate+1.96*std.error),
    ci.lower=exp(estimate-1.96*std.error),
    significance=case_when(
      p.value < .001 & odds.ratio > 1 ~ sprintf('\u2191***'),
      p.value < .01 & odds.ratio > 1 ~ sprintf('\u2191**'),
      p.value < .05 & odds.ratio > 1 ~ sprintf('\u2191*'),
      p.value < .001 & odds.ratio < 1 ~ sprintf('\u2193***'),
      p.value < .01 & odds.ratio < 1 ~ sprintf('\u2193**'),
      p.value < .05 & odds.ratio < 1 ~ sprintf('\u2193*'),
      .default = ''
    )
  )

write_csv(df_results, 'results/associations-nhanes.csv')

