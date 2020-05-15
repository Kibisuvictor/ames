library(tidymodels)
library(tidymodels)
library(ranger)
library(magrittr)
library(tidyverse)
library(skimr)
library(naniar)
library(janitor)


#loading the dataset
#this is the training dataset from kaggle about predicting the house prices, its similar to the ames 
#housing data set
houses_tr <- read_csv("C:\\Users\\hope\\Desktop\\house-prices-advanced-regression-techniques\\train.csv")
houses_tr<-houses_tr %>% clean_names()

#changing characters to factors
str(houses_tr)
houses_tr <- houses_tr %>% mutate_if(is.character, as.factor)
houses_tr$mo_sold <- as.factor(houses_tr$mo_sold)
houses_tr %>% skim(ms_sub_class)
houses_tr$ms_sub_class <- as.factor(houses_tr$ms_sub_class)


#split the data
houses_split <- houses_tr %>% initial_split(prop = 0.9, strata = sale_price)
ames_train <- training(houses_split)
houses_tr %>% view()
houses_tr$bsm


#validation
ames_cv <- vfold_cv(ames_train, v= 5, strata = sale_price)
ames_cv

#recipe
ames_recipe <- recipe(sale_price ~., data = houses_tr) %>% 
  step_log(sale_price, base = 10) %>%
  step_rm(id,street,utilities,lot_config, land_slope,condition2,roof_matl,exter_cond,bsmt_cond,
          bsmt_fin_type2,bsmt_fin_sf2,heating,low_qual_fin_sf,bsmt_half_bath,garage_qual,garage_cond,
          x3ssn_porch,screen_porch,pool_area,misc_val,mo_sold,yr_sold,sale_type) %>% 
  step_novel(all_predictors(), -all_numeric()) %>% 
  step_unknown(all_nominal(), new_level = "none") %>% 
  step_other(all_nominal(), other = "infrequent") %>% 
  step_knnimpute(all_predictors(),-all_nominal()) %>%
  step_corr(all_predictors(), -all_nominal()) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_normalize(all_predictors()) 




# specify model
rand_model <- rand_forest(mode = "regression",
                          mtry = tune(),
                          trees = tune(),
                          min_n = tune()
) %>% 
  set_engine("ranger")

#grid
rand_grid <- grid_random(mtry() %>% range_set(c(2,20)),
                         trees() %>% range_set(c(500,1000)),
                         min_n() %>% range_set(c(2,10)),
                         size = 30)
rand_grid

#workflow
rand_wkflow <- workflow() %>% 
  add_model(rand_model) %>% 
  add_recipe(ames_recipe)

#tune
library(tune)
doParallel::registerDoParallel()
rand_tune <- tune_grid(rand_wkflow,
                       resamples = ames_cv,
                       grid = rand_grid,
                       metrics = metric_set(rmse, rsq),
                       control = control_grid(save_pred = TRUE)
)
rand_tune %>% pluck(".notes")

