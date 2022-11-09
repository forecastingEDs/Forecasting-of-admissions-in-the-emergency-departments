### To use the modeltime package in Rstudio it is necessary to load some Python functions (we use Rstudio loading some functions that are from Python). Run step 1.
### step 1
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
remove.packages("modeltime", lib="~/R/win-library/4.1") # remove the package, if installed.
install.packages("modeltime", dependencies = TRUE)
remotes::install_github("business-science/modeltime", dependencies = TRUE)
# Enter one or more numbers, or an empty line to skip updates: 2 # select option 2: CRAN packages only #

### step 2
### Install and load the following R packages ----

library(keras) 
library(ForecastTB) 
library(ggplot2)
library(zoo)
library(forecast)
library(lmtest)
library(urca)
library(stats)
library(nnfor)
library(forecastHybrid)
library(pastecs)
library(forecastML)
library(Rcpp)
library(modeltime.ensemble)
library(tidymodels)
library(modeltime)
library(lubridate)
library(tidyverse)
library(tidymodels)
library(modeltime.resample)
library(timetk)
library(tidyquant)
library(modeltime.h2o)
library(yardstick)
library(reshape)
library(plotly)
library(xgboost)
library(rsample)
library(targets)
library(modeltime.gluonts)
library(tidymodels)
library(modeltime)
library(modeltime.resample)
library(timetk)
library(tidyverse)
library(tidyquant)
library(LiblineaR)
library(parsnip)
library(ranger)
library(kknn)
library(readxl)
library(skimr) 

### Load Data ----
### In this step you must import the data from the Excel spreadsheet to Rstudio keeping the original file name "atends_temperature_calendar" so that the next steps work correctly.
With Rstudio open, in the upper right corner you have the option/tab >import Dataset>from excel>Browse and select the "atends_temperature_calendar"

### step 3
### Cross validated forecasting using calendar and meteorologists variables ----

data_tbl <- atends_temperature_calendar %>%
  select(id, Date, attendences, average_temperature, min, max, monthday,  sunday, monday, tuesday, wednesday, thursday, friday, saturday, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec) %>%
  set_names(c("id", "date", "value","temperature", "tempemin", "tempemax", "monthday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))

data_tbl

# Full = Training + Forecast Datasets
full_data_tbl <- atends_temperature_calendar %>%
  select(id, Date, attendences, average_temperature, min, max, monthday,  sunday, monday, tuesday, wednesday, thursday, friday, saturday, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec) %>%
  set_names(c("id", "date", "value","temperature", "tempemin", "tempemax", "monthday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")) %>%
  
### Apply Group-wise Time Series Manipulations
  group_by(id) %>%
  future_frame(
    .date_var   = date,
    .length_out = "7 days", # To change the forecast horizon and redo the forecasts at a new horizon, you must change to 3 or 30 days ahead the value in that line of code and redo/rerun the script for that part.
    .bind_data  = TRUE
  ) %>%
  ungroup() %>%
  
  ### Consolidate IDs
  mutate(id = fct_drop(id))

### Training Data
data_prepared_tbl <- full_data_tbl %>%
  filter(!is.na(value))

### Forecast Data
future_tbl <- full_data_tbl %>%
  filter(is.na(value))

data_prepared_tbl %>% glimpse()

### Summary Diagnostics. Let us check the regularity of all time series with timetk::tk_summary_diagnostics()
### Check for summary of timeseries data for training set
data_prepared_tbl %>%
  group_by(id) %>%
  tk_summary_diagnostics()

### step 4
### Data Splitting ----

### Now we set aside the future data (we would only need that later when we make forecast)
### And focus on training data
### * 4.1 Panel Data Splitting ----
### Split the dataset into analyis/assessment set

emergency_tscv <- data_prepared_tbl %>%
  time_series_cv(
    date_var    = date, 
    assess      = "7 days",
    skip        = "60 days",
    cumulative  = TRUE,
    slice_limit = 5
  )

emergency_tscv

emergency_tscv %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, value, 
                           .facet_ncol = 2, .interactive = F)

recipe_spec <- recipe(value ~ ., 
                      data = training(emergency_tscv$splits[[1]])) %>%
  step_timeseries_signature(date) %>%
  step_rm(matches("(.iso$)|(.xts$)|(day)|(hour)|(minute)|(second)|(am.pm)")) %>%
  step_mutate(data = factor(value, ordered = TRUE)) %>%
  step_dummy(all_nominal(), one_hot = TRUE)

summary(recipe_spec)
recipe_spec

summary(prep(recipe_spec))

### Now a "DESIGN MATRIX" is made
recipe_spec %>% prep() %>% juice() %>% glimpse()
recipe_spec %>% prep() %>% juice()

### step 5
### Training of the nine forecast models ----

### Machine learning models
###> Radial Basis Function Support Vector Machine
### Model 1: SVM_rbf ----
wflw_fit_svm_rbf <- workflow() %>%
  add_model(
    svm_rbf("regression") %>% set_engine("kernlab") 
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))

### Model 2: Xgboost ----
wflw_fit_xgboost <- workflow() %>%
  add_model(
    boost_tree("regression") %>% set_engine("xgboost") 
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))

### View the model training workflow from the resulting predictors #
wflw_fit_xgboost
wflw_fit_xgboost %>% 
  extract_fit_parsnip()
wflw_fit_xgboost %>% 
  extract_recipe()

### Model 3: Random Forest ----
wflw_fit_rf <- workflow() %>%
  add_model(
    rand_forest("regression", trees = 500, min_n = 50) %>% set_engine("randomForest")
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>% # Add preprocessing steps (Note that "date" column is removed since Machine Learning algorithms don't typically know how to deal with date or date-time features)
  fit(training(emergency_tscv$splits[[1]]))

### Model 4: ANN ---- 
wflw_fit_ANN <- workflow() %>% 
  add_model(
    nnetar_reg() %>%
      set_engine("nnetar", MaxNWts = 8000)) %>%
  add_recipe(recipe_spec) %>%
  fit(training(emergency_tscv$splits[[1]]))

### Hybrid Models ----
### The Prophet Boost algorithm combines Prophet with XGBoost to get the best of both worlds (i.e. Prophet Automation + Machine Learning). 
### Model 5: Prophet_boost ----
wflw_fit_prophet_boost <- workflow() %>%
  add_model(
    prophet_boost("regression",
                  seasonality_daily  = FALSE, 
                  seasonality_weekly = FALSE,
                  seasonality_yearly = FALSE
    ) %>% 
      set_engine("prophet_xgboost") 
  ) %>%
  add_recipe(recipe_spec) %>%
  fit(training(emergency_tscv$splits[[1]]))

###  Model 6: STLM-ETS ----
wflw_fit_stlm_ets <- workflow() %>%
  add_model(seasonal_reg() %>%
              set_engine(engine = "stlm_ets")
  ) %>%
  add_recipe(recipe_spec) %>%
  fit(training(emergency_tscv$splits[[1]]))

### Statistical models ----

### Model 7: ets ----
wflw_fit_ets <- workflow() %>% 
  add_model(
    exp_smoothing() %>%
      set_engine(engine = "ets")) %>%
  add_recipe(recipe_spec) %>%
  fit(training(emergency_tscv$splits[[1]]))

### ---- NAIVE ----
### Model 8: Naive ----
wflw_fit_naive <- workflow() %>% 
  add_model(
    naive_reg() %>%
      set_engine(engine = "naive")) %>%
  add_recipe(recipe_spec) %>%
  fit(training(emergency_tscv$splits[[1]]))

### Model 9: sNaive ----
wflw_fit_SNAIVE <- workflow() %>% 
  add_model(
    naive_reg() %>%
      set_engine(engine = "snaive")) %>%
  add_recipe(recipe_spec) %>%
  fit(training(emergency_tscv$splits[[1]]))

### step 6
### Create Modeltime Table ----
### The Modeltime Table organizes the models with IDs and creates generic descriptions to help us keep track of our models. 
### Let's add the models to a modeltime_table()
### table modeltime ----

model_tbl <- modeltime_table(
  wflw_fit_xgboost,
  wflw_fit_rf,
  wflw_fit_ANN,
  wflw_fit_svm_rbf,
  wflw_fit_prophet_boost,
  wflw_fit_stlm_ets,
  wflw_fit_ets,
  wflw_fit_naive,
  wflw_fit_SNAIVE
)

model_tbl

### step 7
### Time series cross-validation ---- 3.4 Cross-Validation in Time Series of visits in EDs

resample_results <- model_tbl %>%
  modeltime_fit_resamples(
    resamples = emergency_tscv,
    control   = control_resamples(allow_par = TRUE, verbose = TRUE)
  )

resample_results


### Results ----
resample_results %>%
  plot_modeltime_resamples(yardstick::metric_set(mape, smape, mase, rmse),
                           .point_size  = 4,
                           .summary_line_size  =  1,
                           .point_alpha = 0.8,
                           .interactive = FALSE,
                           .title =  "",
                           .color_lab =  "Models"
  )

resample_results %>%
  plot_modeltime_resamples(yardstick::metric_set(mape, smape, mase, rmse),
                           .point_size  = 3,
                           .summary_line_size  =  1,
                           .point_alpha = 0.8,
                           .title =  "Resample Accuracy",
                           .color_lab =  "Models"
  )

resample_results %>%
  modeltime_resample_accuracy(summary_fns = mean, yardstick::metric_set(mape, smape, mase, rmse)) %>%
  table_modeltime_accuracy(.interactive = FALSE)

### Resample Accuracy Table
### We can get an interactive or static table using modeltime_resample_accuracy(). I'm interested not only in the average metric value but also in the variability (standard deviation). I can get both of these by adding multiple summary functions using a list().

resample_results %>%
  modeltime_resample_accuracy(summary_fns = list(mean = mean, sd = sd), yardstick::metric_set(mape, smape, mase, rmse)) %>%
  table_modeltime_accuracy(.interactive = FALSE)

resample_results %>%
  modeltime_resample_accuracy(summary_fns = mean, yardstick::metric_set(mape, smape, mase, rmse)) %>%
  table_modeltime_accuracy()
