
#--------------------------------------------------------------------------------------------------------------------------
#                                           INITIALIZATION      
#--------------------------------------------------------------------------------------------------------------------------
setwd("C:/Users/dell/Documents")

# Load packages
require(readr)        # loading the csv
require(ggplot2)      # for data exploration
require(tidyr)        # for data manipulation
require(lubridate)    # for date manipulation
require(dplyr)        # for data manipulation
require(caret)        # for train and test partitioning
require(randomForest) # for rf modelling
require(fastAdaboost) # for adaboost modelling
require(ROCR)         # for AUC computation


# Load data set
data =  as.data.frame(read_csv("~/signup_data.csv", col_names = FALSE))

# Label each column
column_names = c("signup_data_id", 
                   "country",
                   "language",
                   "created_at",
                   "email_provider",
                   "is_email_confirmed",
                   "is_privacy_policy_accepted",
                   "is_listed_in_directory",
                   "is_visible",
                   "last_login_at",
                   "site_currency",
                   "google_analytics_activated_at",
                   "user_city",
                   "package_id",
                   "is_blocked")

colnames(data) = column_names

# Initial look on the data
str(data)   # # We have 200,000 observations and 15 columns and features are integers, character strings and dates

# Transform each column into proper format for easier exploratory data analysis
data$country = as.factor(data$country)
data$language = as.factor(data$language)
data$email_provider = as.factor(data$email_provider)
data$is_email_confirmed = as.factor(data$is_email_confirmed)
data$is_listed_in_directory = as.factor(data$is_listed_in_directory)
data$is_visible = as.factor(data$is_visible)
data$site_currency = as.factor(data$site_currency)
data$user_city = as.factor(data$user_city)
data$package_id = as.factor(data$package_id)
data$is_blocked = as.factor(data$is_blocked)

# Create a new column which translates is_blocked column into a spam if true and ham otherwise
data$class =  as.factor(ifelse(data$is_blocked == "true", c("spam"), c("ham")))


#--------------------------------------------------------------------------------------------------------------------------
#                                           EXPLORATORY      
#--------------------------------------------------------------------------------------------------------------------------
summary(data) # We can see from the summary that we have a number of missing values on different columns.

####  Now let's investigate each column
# Check consistency of values/levels per column specifically type character columns
# This will verify existence of encoding errors if there are any and counts of each level.


# signup_data_id
sort(table(data$signup_data_id), decreasing = T)[1:10]    # There ID's with multiple sign ups.

# Country
plot(sort(table(data$country), decreasing = T))  # We have a lot of levels but only a few makes up majority of the data.
sort(table(data$country), decreasing = T)[1:10]  # Showing the top 10 countries
table(data$country, data$class)                  # High occurence of spam on countries like US, RU, DO, DE

# Language
sort(table(data$language), decreasing = T)
plot(sort(table(data$language), decreasing = T))  # Major language are en, de and eslanguage, we can treat the last 7 items as "others".
table(data$language, data$class)                  # Occurence of spam per language
ggplot(data) + geom_bar(aes(x = language, fill = class), position = "fill")        
# We can see that users using english, russian, turkish and french language are
# have higher occurence of spam as compared to other languages


# email_provider
# First we need to extract the email main provider from the complete email provider description
data$email_provider = as.character(data$email_provider)
data = separate(data, email_provider,into = c("email_main_provider", "email_extension1","email_extension2"), 
                sep = "\\.", remove = T)
sort(table(data$email_main_provider),decreasing = T)[1:10]       # Looking on the first 10 show that gmail and yahoo are the top email providers.
cols = names(data) %in% c("email_extension1","email_extension2") # index the 2 columns as TRUE
data = data[!cols]                                               # remove the 2 columns from data


# is_email_confirmed
table(data$is_email_confirmed)
ggplot(data) + geom_bar(aes(x = is_email_confirmed, fill = class), position = "fill") # is_email_confirmed vs class


# is_privacy_policy_accepted
table(data$is_privacy_policy_accepted)   # Having 1 value for this column adds no value in the model.


# is_listed_in_directory
table(data$is_listed_in_directory)       # Having 1 value for this column adds no value in the model


# is_visible
table(data$is_visible)  
ggplot(data) + geom_bar(aes(x = is_visible, fill = class), position = "fill")         

# site_currency
table(data$site_currency)                   # This column is completely blank
sum(is.na(data$site_currency))

# user_City
sort(table(data$user_city), decreasing = T)[1:10]   # Showing only the top 10 but has too many levels, need to trim down.

# package_id
table(data$package_id)
ggplot(data) + geom_bar(aes(x = package_id, fill = class), position = "fill")      # Higher occurence of spam if  package = FREE


#--------------------------------------------------------------------------------------------------------------------------
#                                           INSPECTION      
#--------------------------------------------------------------------------------------------------------------------------
# Create a function computing the number and percentage of missing data per column
count_na = function(y){
  count  = sapply(y, function(y) sum(length(which(is.na(y)))))
  count  = data.frame(count, percentage = (count/nrow(y)*100))
  return (count)
}

count_na(data)

#------------------------------------DEALING WITH NAS  -----------------------------------------------------------
#
# What to do with columns that has NA's and columns that will have no contribution in the model?
#
# is_privacy_policy_accepted    > remove variable since it only has one value which is "false"
# is_listed_in_directory        > remove variable since it only has one value which "false"
# site_currency                 > remove variable since there is no information available, 100% NA
# google_analytics_activated_at > remove variable, since it has only 5% completeness, cannot impute dates either.
# last_login_at                 > delete rows with NAs. Cannot impute nor create a new 
#                                 categorical like "no_login" because it will distort the existing variable type (date with chr)
# country                       > create a new category, "unknown"
# user_city                     > create a new category, "unknown"
#-----------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------
#                                 DATA CLEANING AND PREPARATION FOR MODELLING  
#--------------------------------------------------------------------------------------------------------------------------

# Remove columns that won't be helpful in the analysis
cols_to_remove = c("is_privacy_policy_accepted", 
                   "is_listed_in_directory", 
                   "site_currency",
                   "google_analytics_activated_at")

# Create a logical vector indicating whether columns should be removed "TRUE" or not "FALSE 
cols = names(data) %in% cols_to_remove

# Remove the columns from the data and resave the trimmed data as data
data = data [!cols]

# Delete rows with NA last_login_at, cannot impute date values anyway
data = data[which(!is.na(data$last_login_at)),]

# Create an "unknown" category for missing country
data$country = as.character(data$country)
data[which(is.na(data$country)),"country"] = "unknown"


# We still need to deal with the categorical variables with significantly high number of levels
# We need to specify the number of levels we need to treat as major categories so that the rest can be categorized as "others" or "unknown"
# Note that we need to limit it to <= 32 for the random forest
nc = 20     # number of major countries (played around 20 and 25 but same model performance)
nl = 8      # number of major language
ne = 15     # number of major email_provider(played around 30 and 15 but same model performance)

# country
# Create ""others" category for country 
country_rankings = sort(table(data$country), decreasing = T)
sum(country_rankings[1:nc])/sum(country_rankings)   # nc countries represents 82% of the total number of observations
major_countries = names(country_rankings[1:nc])     # Major countries will retain their labels.

data$country_grouped = data$country                 # Create a new column to preserve the old one
# Classiffy those countries NOT belonging to major_countries or "unknown" countries as "others"
data$country_grouped[-which(data$country %in% major_countries | data$country == "unknown")] = "others"
data$country_grouped = as.factor(data$country_grouped)


# language
# Create ""others" category for language
data$language = as.character(data$language)
language_rankings = sort(table(data$language), decreasing = T)
sum(language_rankings[1:nl])/sum(language_rankings)   # Comprises 99.4% of the observation 
major_language = names(language_rankings[1:nl])

data$language_grouped = data$language                 # Create a new column to preserve the old one
# Classiffy those language NOT belonging to major_language  as "others"
data$language_grouped[-which(data$language %in% major_language)] = "others"
data$language_grouped = as.factor(data$language_grouped)

# user_city : We cannot do much about user_city since majority of the signups has unknown city.
#             Instead of deleting the whole column, the best that we can do is to 
#             dichotomize the user_city into "unknown" and "known"
data$user_city = ifelse(is.na(data$user_city), "unknown","known")
data$user_city = as.factor(data$user_city)
table(data$user_city, data$class)

# email_provider : Create others" category for non major email providers
email_provider_rankings = sort(table(data$email_main_provider), decreasing = T)
sum(email_provider_rankings[1:ne])/sum(email_provider_rankings)   # Comprises 81.6% of the tota number of observation
major_email_providers = names(email_provider_rankings[1:ne])      # Tag them as major email_providers

data$email_main_provider_grouped = data$email_main_provider       # Create a new column to preserve the old one
# Classiffy email providers NOT belonging to major_email_providers as "others"
data$email_main_provider_grouped[-which(data$email_main_provider %in% major_email_providers)] = "others"
data$email_main_provider_grouped = as.factor(data$email_main_provider_grouped)


# Feature Engineering
# Create new features out of created_at and last_login_at column
data = data %>% 
  mutate(
    created_at_date = date(data$created_at),
    created_at_day = wday(data$created_at, label = TRUE),
    created_at_hour = hour(data$created_at),
    
    last_login_at_date = date(data$last_login_at),
    last_login_at_day  = wday(data$last_login_at, label = TRUE),
    last_login_at_hour = hour(data$last_login_at)
  ) %>% 
  select(-last_login_at, -created_at)    # remove original columns


# Explore the cleaned data set and see if there are still any anomalies present.
str(data)
summary(data)

# We can now extract our set of predictors to form the final data for modelling.
columns = c("class",
            "is_email_confirmed",
            "is_visible",
            "user_city",
            "package_id",
            "country_grouped",
            "language_grouped",
            "email_main_provider_grouped",
            "created_at_date",
            "created_at_day",
            "created_at_hour",
            "last_login_at_date",
            "last_login_at_day",
            "last_login_at_hour"
)
final_data = data[columns]  

# Split the data set into Train and Test set.
# Goal: Create around 80/20 split between the train and test set
# Note: Train and test set should reflect the same distribution between ham and spam

# Extract 80% of the rows as the train set leaving the 20% to be the test set.
set.seed(10)
train_index =  createDataPartition(final_data$class, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
train_set = final_data[train_index,]
test_set = final_data[-train_index,]

# Let's check if the distribution between spam and ham is preserved in the train and test set.
table(final_data$class)/nrow(final_data)
table(train_set$class)/nrow(train_set)
table(test_set$class)/nrow(test_set)


#--------------------------------------------------------------------------------------------------------------------------
#                                       MODELLING : RANDOM FOREST
#--------------------------------------------------------------------------------------------------------------------------


# Scenario 1: Completely ignore the predictors that have > 32 levels originally like country, email_main_provider
#             which are now grouped in our train set.
set.seed(11)
rf1_model = randomForest(class~. -email_main_provider_grouped -country_grouped, 
                         data = train_set, ntree = 500,
                         classwt = c(0.913, 0.087))
rf1_prediction_class = predict(rf1_model, test_set, type = "response")
rf1_conf_matrix = as.matrix(table(test_set$class,rf1_prediction_class))
rf1_accuracy = sum(diag(rf1_conf_matrix))/sum(rf1_conf_matrix) # but since the target variable is imbalance, we cannot rely on accuracy alone
rf1_precision = rf1_conf_matrix[2,2] / (rf1_conf_matrix[2,2] + rf1_conf_matrix[1,2])    # TP/(TP+FP)
rf1_recall = rf1_conf_matrix[2,2] / (rf1_conf_matrix[2,2] + rf1_conf_matrix[2,1])       # TP/(TP+FN)

# Plot the ROC curve and Compute for AUC
rf1_prediction_prob = predict(rf1_model, test_set, type = "prob" )
rf1_eval = prediction(rf1_prediction_prob[,2],test_set$class)
plot(performance(rf1_eval,"tpr","fpr"))
rf1_auc = attributes(performance(rf1_eval,'auc'))$y.values[[1]]

# Results
rf1_conf_matrix  #      prediction_class
#                                 ham spam
#                           ham  35023   334
#                           spam   2231  1140
rf1_accuracy     #      0.9337688
rf1_precision    #      0.7734057
rf1_recall       #      0.3381786
rf1_auc          #      0.8213785


#--------------------------------------------------variable importance-----------------------------------------------------------------

# Before we try further any use cases for the random forest model, we need to  identify what are the important variables.
set.seed(12)
rf_importance = randomForest(class~., data = train_set, ntree = 500, classwt = c(0.913, 0.087),importance = T)
varImpPlot(rf_importance)
#-----------------------------------------------------------------------------------------------------------------------------------

# Scenario 2: What if we try a forward selection of predictors starting from the most important variable 
#             then select the optimal model as final random forest model (optimall means smaller number of predictors yet high AUC value)

# Arrange the predictors based on importance, this will be the order of predictors entering the model
sorted = names(sort(importance(rf_importance)[,c(3)], decreasing = T))

# Let's split further our train set into small train set and validation set for model selection.
set.seed(13)
small_train_index = createDataPartition(train_set$class, p = 0.90, list = FALSE, times = 1)

small_train_set = train_set[small_train_index,]
validation_set = train_set[-small_train_index,]

# initialize containers for the result
rf2_model = list()
rf2_prediction_class = as.data.frame(matrix(0, nrow(validation_set), length(sorted)))
rf2_prediction_prob = as.data.frame(matrix(0, nrow(validation_set), length(sorted)))
rf2_eval = list()
rf2_auc = list()

# Forward selection of predictors
set.seed(14)
for (i in 1 : length(sorted)) { 
  rf2_model[[i]] = randomForest(class~., data = small_train_set[,c("class", sorted[1:i])], classwt = c(0.913, 0.087), ntree = 500) 
  rf2_prediction_class[,i] = predict(rf2_model[[i]],validation_set, type = "response")   
  rf2_prediction_prob[,i] = predict(rf2_model[[i]],validation_set, type = "prob" )
  rf2_eval[[i]] = prediction(rf2_prediction_prob[[i]][,2],validation_set$class)
  rf2_auc[[i]] = attributes(performance(rf2_eval[[i]],'auc'))$y.values[[1]]
}

plot(c(1:length(sorted)), rf2_auc) 
# we can see that we can create a simpler model but delivering the same performance as one with 13 predictors
# this also validates the result of the variable importance conducted

# Now lets' build our final model using the complete train_set and selected predictors
final_predictors  = sorted[1:8]      # based on the plot We consider up to the top 8 important predictors

# final model has the following predictors
#  "email_main_provider_grouped" 
# "last_login_at_date"          
# "created_at_date"   
#  "country_grouped"             
# "language_grouped"  
# "is_email_confirmed"     
#  "is_visible"                 
# "created_at_hour"   
# Majority of the most important predictors are either re-coded or generated feature out of the existing feature.

# Final Model
set.seed(15)
final_rf_model = randomForest(class~., data = train_set[,c("class", final_predictors)], classwt = c(0.913, 0.087), ntree = 500)
final_rf_prediction_class = predict(final_rf_model, test_set, type = "response")
final_rf_conf_matrix = as.matrix(table(test_set$class,final_rf_prediction_class))
final_rf_accuracy = sum(diag(final_rf_conf_matrix))/sum(final_rf_conf_matrix)  # but since the target variable is imbalance, we cannot rely on accuracy alone
final_rf_precision = final_rf_conf_matrix[2,2] / (final_rf_conf_matrix[2,2] + final_rf_conf_matrix[1,2])    # TP/(TP+FP)
final_rf_recall = final_rf_conf_matrix[2,2] / (final_rf_conf_matrix[2,2] + final_rf_conf_matrix[2,1])       # TP/(TP+FN)

# Plot the ROC curve and Compute for AUC
final_rf_prediction_prob = predict(final_rf_model, test_set, type = "prob" )
final_rf_eval = prediction(final_rf_prediction_prob[,2],test_set$class)
plot(performance(final_rf_eval,"tpr","fpr"))
final_rf_auc = attributes(performance(final_rf_eval,'auc'))$y.values[[1]]

# Final Random Forest Results
final_rf_conf_matrix #           prediction_class
                     #              ham spam
                     #       ham  34970   387
                     #       spam 1577  1794
final_rf_accuracy    #  0.9492873
final_rf_precision   #  0.8225585
final_rf_recall      #  0.5321863
final_rf_auc         #  0.9080597


#--------------------------------------------------------------------------------------------------------------------------
#                                       MODELLING : ADABOOST
#--------------------------------------------------------------------------------------------------------------------------

# Scenario 1: Considering all predictors

set.seed(16)
adaboost1_model = real_adaboost(class ~., data = train_set, nIter = 100)
adaboost1_prediction = predict(adaboost1_model,test_set)
adaboost1_conf_matrix = as.matrix(table(test_set$class, adaboost1_prediction$class))
adaboost1_accuracy = sum(diag(adaboost1_conf_matrix))/sum(adaboost1_conf_matrix)  # but since the target variable is imbalance, we cannot rely on accuracy alone
adaboost1_precision = adaboost1_conf_matrix[2,2] / (adaboost1_conf_matrix[2,2] + adaboost1_conf_matrix[1,2])    # TP/(TP+FP)
adaboost1_recall = adaboost1_conf_matrix[2,2] / (adaboost1_conf_matrix[2,2] + adaboost1_conf_matrix[2,1])  # TP/(TP+FN)
adaboost1_eval = prediction(adaboost1_prediction$prob[,2],test_set$class)
plot(performance(adaboost1_eval,"tpr","fpr"))
adaboost1_auc = attributes(performance(adaboost1_eval,'auc'))$y.values[[1]]



# adaboost1 Results
adaboost1_conf_matrix #   prediction_class
                      #           ham spam
                      #      ham  34784   573
                      #      spam  1659  1712

adaboost1_accuracy    # 0.9423673
adaboost1_precision   # 0.7492341
adaboost1_recall      # 0.5078612
adaboost1_auc         # 0.5910375



# Scenario 2: Considering only the important predictors

set.seed(17)  
adaboost2_model = real_adaboost(class ~., data = train_set[,c("class", final_predictors)], nIter = 100)
adaboost2_prediction = predict(adaboost2_model,test_set)
adaboost2_conf_matrix = as.matrix(table(test_set$class, adaboost2_prediction$class))
adaboost2_accuracy = sum(diag(adaboost2_conf_matrix))/sum(adaboost2_conf_matrix)  # but since the target variable is imbalance, we cannot rely on accuracy alone
adaboost2_precision = adaboost2_conf_matrix[2,2] / (adaboost2_conf_matrix[2,2] + adaboost2_conf_matrix[1,2])    # TP/(TP+FP)
adaboost2_recall = adaboost2_conf_matrix[2,2] / (adaboost2_conf_matrix[2,2] + adaboost2_conf_matrix[2,1])       # TP/(TP+FN)
adaboost2_eval = prediction(adaboost2_prediction$prob[,2],test_set$class)
plot(performance(adaboost2_eval,"tpr","fpr"))
adaboost2_auc = attributes(performance(adaboost2_eval,'auc'))$y.values[[1]]



# adaboost2 Results
adaboost2_conf_matrix #   prediction_class
                      #           ham  spam
                      #      ham  34673   684
                      #      spam  1561  1810

adaboost2_accuracy    # 0.9420316
adaboost2_precision   # 0.7257418
adaboost2_recall      # 0.5369327
adaboost2_auc         # 0.5037904


