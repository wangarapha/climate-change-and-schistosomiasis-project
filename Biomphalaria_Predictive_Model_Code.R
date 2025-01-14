# R code for Biomphalaria spp. habitata suitablilty analysis and prediction in the Lake Victoria Basin (LVB)

# Climate change drives small scale changes in snail habitat suitability and Schistosomiasis Risk in the Lake Victoria Basin in Sub-Saharan Africa

#-------------------------------------------------Load necessary libraries-------------------------------------------
# Load necessary libraries
library(terra)         # For handling spatial and raster data
library(tidyverse)     # For data manipulation and visualization
library(dismo)         # For species distribution modeling
library(raster)        # For raster data manipulation
library(randomForest)  # For random forest modeling
library(xgboost)       # For extreme gradient boosting
library(mlr3)          # For machine learning workflows
library(maxnet)        # For MaxEnt species distribution modeling
library(pROC)          # For ROC curve analysis
library(caret)         # For training and testing machine learning models
library(biomod2)       # For ensemble species distribution modeling
library(ggplot2)       # For advanced visualization
library(gridExtra)     # For arranging multiple plots on a grid

#--------------------------------------Set working directories--------------------------------------------------
# Set working directories
occurrence_points_path <- "C:/Users/DELL/Documents/Schist_Points/Biom_Points.csv"
env_variables_path <- "F:/Current_EVs"

#--------------------------------Load occurrence data and enverionmental variables data-------------------------
# Load occurrence points
occurrence <- read.csv(occurrence_points_path)
occurrence <- occurrence[, c("longitude", "latitude")]

#Load all the environmental adata raster files
env_files <- list.files(env_variables_path, pattern = "\\.tif$", full.names = TRUE)

#------------------------------------------ Align and resample environmental raster files ------------------------ 
# set the first raster as a reference for extent and resolution
ref_raster <- rast(env_files[1])
# Function to check extents and reproject if necessary
check_and_align_raster <- function(raster, ref_raster) {
  # Check if CRS matches
  if (!same.crs(raster, ref_raster)) {
    raster <- project(raster, crs(ref_raster))
    cat("Reprojected raster to match CRS of the reference raster.\n")
  }
  
  # Check if extents overlap using `intersect()` for extents
  if (is.null(intersect(ext(raster), ext(ref_raster)))) {
    stop("Raster extents do not overlap.")
  }
  
  # Crop and resample the raster to match the reference raster
  raster_cropped <- crop(raster, ext(ref_raster))
  raster_resampled <- terra::resample(raster_cropped, ref_raster, method = "bilinear")
  
  return(raster_resampled)
}

# Apply the function to each raster file
aligned_rasters <- lapply(env_files, function(file) {
  raster <- rast(file)
  check_and_align_raster(raster, ref_raster)
})

# Stack the aligned rasters
env_layers <- rast(aligned_rasters)

# Check the final stack
print(env_layers)
############### Convert occurrence data to a SpatVector and align with environmental layers ################
#Ensure occurrence data is a data frame with longitude  and latitude  columns:
# Convert occurrence data to a SpatVector
occ_vector <- vect(occurrence, geom = c("longitude", "latitude"), crs = crs(env_layers))
print(occ_vector)

# Ensure CRS alignment between occurrence points and environmental layers
if (!same.crs(occ_vector, env_layers)) {
  occ_vector <- project(occ_vector, crs(env_layers))
  cat("Reprojected occurrence points to match CRS of the environmental layers.\n")
}

#-----------------------------------------------Generate Pseudoabsence points----------------------------------
# Function to generate pseudoabsence points using the surface-range envelope method
# Extract environmental data for presence points
  occ_env_data <- extract(env_layers, occ_vector)
  occ_env_data <- na.omit(occ_env_data)  # Remove NA values
  print(head(occ_env_data))  # Verify the extracted data
  
# Calculate quantile range (e.g., 2.5% to 95%) for each environmental variable
  quantile_range <- c(0.025, 0.95)
  lower_bounds <- apply(occ_env_data, 2, quantile, probs = quantile_range[1], na.rm = TRUE)
  upper_bounds <- apply(occ_env_data, 2, quantile, probs = quantile_range[2], na.rm = TRUE)
  print(lower_bounds)  # Check the lower bounds
  print(upper_bounds)  # Check the upper bounds
  
# Identify suitable areas based on the environmental thresholds
  suitable_area <- env_layers
  for (var in names(env_layers)) {
    suitable_area[[var]] <- (env_layers[[var]] >= lower_bounds[var]) & (env_layers[[var]] <= upper_bounds[var])
  }
  
# Combine all suitable areas to create a binary "suitable mask"
  combined_suitable_area <- app(suitable_area, all, na.rm = TRUE)
  
# Invert the mask to define unsuitable areas
  unsuitable_area <- combined_suitable_area == 0
  plot(unsuitable_area, main = "Unsuitable Area")
  
# Generate random points within the unsuitable area
  pseudoabsence_points <- spatSample(unsuitable_area, size = 500, method = "random", na.rm = TRUE, xy = TRUE)
  
  Print (pseudoabsence_points)
  
# Convert pseudoabsence points to a SpatVector
  pseudoabsence_vector <- vect(pseudoabsence_points, geom = c("x", "y"), crs = crs(env_layers))
  
# Visualize occurrence and pseudoabsence points
plot(env_layers[[1]], main = "Occurrence and Pseudoabsence Points")
points(occ_vector, col = "blue", pch = 20, cex = 1.5)
points(pseudoabsence_points, col = "red", pch = 20, cex = 1.5)
print(pseudoabsence_points)

#----------------------------------Combine presence and pseudoabsence data----------------------------------
# Create the presence data frame from occurrence points with lat, lon and presence = 1 column
presence_data <- data.frame(lon = occurrence$longitude, lat = occurrence$latitude, presence = 1)
# Check the presence data frame
print(presence_data)

# Convert pseudoabsence_points to a data frame
pseudoabsence_df <- as.data.frame(pseudoabsence_points)
# Remove the 'all' column to retain only the x and y columns and remove everything
pseudoabsence_df <- pseudoabsence_df[, c("x", "y")]
# Create the pseudoabsence data frame with lat, lon and presence = 0 column
pseudoabsence_data <- data.frame(lon = pseudoabsence_df$x, lat = pseudoabsence_df$y, presence = 0)
# Check the pseudoabsence data frame
print(pseudoabsence_data)

# Combine presence and pseudoabsence data
presence_absence <- rbind(presence_data, pseudoabsence_data)
print(presence_absence)

# Visualization of presence_absence data
presence_absence %>%
  group_by(presence) %>% 
  ggplot(aes(x=lon, y=lat, fill=presence, colour = presence))+
  geom_point(alpha=1, )+
  theme_bw()

# -------------------------------Prepare data for XGBoost, RF and Maxent, model runs--------------------------- 

# Conditional function to handle NA values in raster layers
handle_na <- function(presence_absence, env_layers) {
  presence_absence <- na.omit(presence_absence)  # Remove rows with NA values
  return(presence_absence)
}

# Create a function to evaluate models and calculate AUC
evaluate_model <- function(model, test_data, predictions) {
  auc_val <- roc(test_data$presence, predictions)$auc
  return(auc_val)
}

###################################### 1. XGBoost Model Trainin & Analysis ##################################
# Initialize lists to store results
aucs <- c()
rasters <- list(XGBoost = list())
model_results <- list(XGBoost = list())

# XGBoost Model Analysis with xgb.cv
for (i in 1:25) {
  
  # Set a seed for reproducibility and ensure different splits
  set.seed(i)
  
  # Split data into training and testing sets (training (80%) and testing (20%))
  train_indices <- sample(1:nrow(presence_absence), size = round(0.8 * nrow(presence_absence)))
  train_data <- presence_absence[train_indices, ]
  test_data <- presence_absence[-train_indices, ]
  
  # Extract the environmental data for the training data based on lon and lat
  env_train_data <- terra::extract(env_layers, train_data[, c("lon", "lat")])
  
  # Combine the presence-absence data with the environmental data
  train_data_combined <- cbind(train_data, env_train_data)
  
  # Exclude non-predictor columns (ID, lon, lat) for model fitting
  predictors <- train_data_combined[, !(names(train_data_combined) %in% c("ID", "lon", "lat"))]
  
  # Convert data to DMatrix format required by XGBoost
  train_matrix <- xgboost::xgb.DMatrix(data = as.matrix(predictors[, -1]), label = predictors$presence)
  
  # Perform cross-validation using xgb.cv and hyperparamter tuning 
  cv_results <- xgboost::xgb.cv(
    data = train_matrix,
    nfold = 5,  # Number of cross-validation folds
    nrounds = 500,  # Number of boosting iterations
    early_stopping_rounds = 10,  # Early stopping
    max_depth = 5,  # Depth of trees
    eta = 0.1,  # Learning rate
    objective = "binary:logistic",  # Binary classification
    eval_metric = "auc",  # Evaluation metric
    verbose = TRUE
  )
 
  # Extract the best number of boosting rounds from cross-validation
  best_nrounds <- cv_results$best_iteration
  cat("Best number of boosting rounds:", best_nrounds, "\n")
  
  # Extract the training AUC from cross-validation results
  train_auc <- max(cv_results$evaluation_log$train_auc_mean)
  cat("Training AUC for Iteration", i, ":", round(train_auc, 4), "\n")
  print(train_auc)
  
  # Train the final XGBoost model on the full training data
  xgb_model <- xgboost::xgb.train(
    data = train_matrix,
    nrounds = best_nrounds,
    max_depth = 5,
    eta = 0.1,
    objective = "binary:logistic",
    eval_metric = "auc"
  )
  
  # Extract environmental data for the test set
  env_test_data <- terra::extract(env_layers, test_data[, c("lon", "lat")])
  test_data_combined <- cbind(test_data, env_test_data)
  
  # Exclude non-predictor columns from test data
  test_predictors <- test_data_combined[, !(names(test_data_combined) %in% c("ID", "lon", "lat"))]
  
  # Convert test data to DMatrix format
  test_matrix <- xgboost::xgb.DMatrix(data = as.matrix(test_predictors[, -1]))
  
  # Predict on the test dataset
  xgb_pred <- predict(xgb_model, newdata = test_matrix)
  
  # Evaluate AUC for XGBoost
  auc_xgb <- roc(test_data$presence, xgb_pred)$auc
  aucs <- c(aucs, auc_xgb)
  print(auc_xgb)
  
  # Determine the average AUC for the XGBoost model
  average_xgb_auc <- mean(aucs, na.rm = TRUE)
  print(average_xgb_auc)
  
  # Save the XGBoost model prediction raster
  xgb_raster <- terra::predict(
    env_layers,
    xgb_model,
    fun = function(model, d) {
      predict(model, xgboost::xgb.DMatrix(as.matrix(d)))
    }
  )
  writeRaster(xgb_raster, filename = paste0("xgb_prediction_", i, ".tif"), overwrite = TRUE)
  
  # Store the results
  rasters[["XGBoost"]] <- append(rasters[["XGBoost"]], list(xgb_raster))
  model_results[["XGBoost"]] <- append(model_results[["XGBoost"]], auc_xgb)
}
#################################### 2. Random Forest Model training & analysis ############################
# Initialize lists to store results
model_results <- list(RF = list())
rasters <- list(RF = list())
aucs <- numeric()

# Define cross-validation control
train_control <- trainControl(
  method = "cv",                # Cross-validation
  number = 5,                   # Number of folds
  classProbs = TRUE,            # Enable probability predictions
  summaryFunction = twoClassSummary  # Evaluate using AUC
)

# RF Model Analysis with Cross-Validation
for (i in 1:25) {  # Reduce iterations for demonstration purposes
  # Set a seed for reproducibility
  set.seed(i)
  
  # Split data into training and testing sets (80% training, 20% testing)
  train_indices <- sample(1:nrow(presence_absence), size = round(0.8 * nrow(presence_absence)))
  train_data <- presence_absence[train_indices, ]
  test_data <- presence_absence[-train_indices, ]
  
  # Extract environmental data for the training data based on lon and lat
  env_train_data <- terra::extract(env_layers, train_data[, c("lon", "lat")])
  
  # Combine and clean training data
  train_data_combined <- cbind(train_data, env_train_data)
  train_data_combined <- na.omit(train_data_combined)  # Remove rows with missing values
  
  # Ensure the `presence` column is a factor
  train_data_combined$presence <- factor(train_data_combined$presence, levels = c(0, 1), labels = c("Absent", "Present"))
  
  # Exclude non-predictor columns (ID, lon, lat)
  predictors_train <- train_data_combined[, !(names(train_data_combined) %in% c("ID", "lon", "lat"))]
  
  # Train the Random Forest model with CV using caret
  rf_tuned <- caret::train(
    presence ~ ., 
    data = predictors_train,
    method = "rf",
    tuneLength = 5,               # Auto-select a grid of hyperparameters
    metric = "ROC",               # Use AUC as the performance metric
    trControl = train_control,     # Cross-validation control
    ntree = 500,                              # Specify 500 trees
    importance = TRUE                         # Measure variable importance
  )
  
  # Extract the best model
  rf_model <- rf_tuned$finalModel
  # cat("Best Parameters for Iteration", i, ":", rf_tuned$bestTune, "\n")
 
  # Extract environmental data for testing
  env_test_data <- terra::extract(env_layers, test_data[, c("lon", "lat")])
  test_data_combined <- cbind(test_data, env_test_data)
  test_data_combined <- na.omit(test_data_combined)  # Remove rows with missing values
  
  # Exclude non-predictor columns from test data
  predictors_test <- test_data_combined[, !(names(test_data_combined) %in% c("ID", "lon", "lat"))]
  
  # Predict on the test dataset
  rf_pred <- predict(rf_model, newdata = predictors_test[, -1], type = "prob")[, 2]
  
  # Evaluate AUC for the RF model
  auc_rf <- roc(test_data_combined$presence, rf_pred)$auc
  aucs <- c(aucs, auc_rf)
  print(paste("Iteration", i, "AUC:", auc_rf))
  
  # Save the RF model prediction raster
  rf_raster <- terra::predict(
    env_layers, 
    rf_model, 
    fun = function(model, d) predict(model, d, type = "prob")[, 2],
    na.rm = TRUE  # Ensure missing values are handled
  )
  writeRaster(rf_raster, filename = paste0("rf_prediction_", i, ".tif"), overwrite = TRUE)
  
  # Store results
  rasters[["RF"]] <- append(rasters[["RF"]], list(rf_raster))
  model_results[["RF"]] <- append(model_results[["RF"]], auc_rf)
}
# Determine the average AUC for the rf model
average_rf_auc <- mean(aucs, na.rm = TRUE)
print(average_rf_auc)

# Print final average AUC
cat("Average AUC across iterations:", mean(aucs, na.rm = TRUE), "\n")

##################################### 3. Maxent Model training & analysis ##############################
# Load environmental variables as spatraster
# List all .tif files
tif_files <- list.files(path = "F:/Current_EVs", pattern = "\\.tif$", full.names = TRUE)

# Load rasters individually to check their properties
rasters <- lapply(tif_files, rast)

# Print extent and resolution of each raster for comparison
lapply(rasters, function(x) {
  print(ext(x))
  print(res(x))
})

# Choose a reference raster (e.g., the first one)
reference_raster <- rasters[[1]]

# Align other rasters to the reference raster
aligned_rasters <- lapply(rasters, function(x) {
  if (!compareGeom(x, reference_raster, stopOnError = FALSE)) {
    # Use the terra resample function
    terra::resample(x, reference_raster, method = "bilinear")
  } else {
    x
  }
})

reference_crs <- crs(reference_raster)

reprojected_rasters <- lapply(rasters, function(x) {
  if (!compareGeom(x, reference_raster, stopOnError = FALSE)) {
    project(x, reference_crs)
  } else {
    x
  }
})

aligned_rasters <- lapply(reprojected_rasters, function(x) {
  if (!compareGeom(x, reference_raster, stopOnError = FALSE)) {
    terra::resample(x, reference_raster, method = "bilinear")
  } else {
    x
  }
})

# Combine the aligned rasters into a single SpatRaster
env_layers <- rast(aligned_rasters)

# Load occurrence data (latitude and longitude)
occur_data <- read.csv(occurrence_points_path)

# Ensure occurrence data has latitude and longitude columns
if (!all(c("longitude", "latitude") %in% colnames(occur_data))) {
  stop("Occurrence data must have 'longitude' and 'latitude' columns.")
}

# Ensure occur_data is a data frame and has the required columns
if (!is.data.frame(occur_data)) {
  occur_data <- as.data.frame(occur_data)
}

# Check column names
colnames(occur_data) <- tolower(colnames(occur_data))  # Make sure column names match

# Extract environmental data using 'terra::extract()'
occur_env_data <- terra::extract(env_layers, occur_data[, c("longitude", "latitude")])


# Combine occurrence data with extracted environmental data and filter out NAs
occur_combined <- cbind(occur_data, occur_env_data)
occur_combined <- occur_combined[complete.cases(occur_combined), ]
occur_combined$presence <- 1  # Label presence points

# Generate background points using 'terra'
background_points <- spatSample(env_layers, size = 1000, method = "random", xy = TRUE, na.rm = TRUE)

# Ensure 'background_points' is in the correct format with 'x' and 'y' columns
if (!is.data.frame(background_points)) {
  background_points <- as.data.frame(background_points)
}

# Check column names to ensure they are correct
if (!all(c("x", "y") %in% colnames(background_points))) {
  colnames(background_points) <- c("x", "y")
}

# Extract environmental data using 'terra::extract()'
background_env_data <- terra::extract(env_layers, background_points[, c("x", "y")])

# Check if extraction was successful
if (!is.null(background_env_data) && nrow(background_env_data) > 0) {
  print("Extraction successful.")
} else {
  print("Extraction failed or returned no data.")
}
background_combined <- data.frame(presence = 0, background_points, background_env_data)
background_combined <- background_combined[complete.cases(background_combined), ]

# Identify and remove duplicate environmental columns with '.1' suffix in background_combined
duplicate_columns <- grep("\\.1$", colnames(background_combined), value = TRUE)
background_combined <- background_combined[, !colnames(background_combined) %in% duplicate_columns]

# Standardize column names for 'occur_combined' to match 'background_combined'
colnames(occur_combined)[colnames(occur_combined) %in% c("longitude", "latitude")] <- c("x", "y")

# Combine presence and background data
full_data <- rbind(occur_combined, background_combined)

# Check if the combination was successful
print("Data combined successfully.")

# Determine the response variable
response_var <- full_data$presence

# Create predictor_vars excluding 'id', 'presence', 'x', and 'y' columns
predictor_vars <- full_data[, !(colnames(full_data) %in% c("ID", "presence", "x", "y"))]

# Check for NA values and remove rows with NA in predictor_vars
predictor_vars <- predictor_vars[complete.cases(predictor_vars), ]

# Run MaxEnt model 25 times
for (i in 1:25) {
  tryCatch({
    # Split data into training (80%) and testing (20%)
    set.seed(i)
    
    train_indices <- sample(1:nrow(full_data), size = round(0.8 * nrow(full_data)))
    train_data <- predictor_vars[train_indices, ]
    test_data <- predictor_vars[-train_indices, ]
    train_response <- response_var[train_indices]
    test_response <- response_var[-train_indices]
    
    # Train the maxent model
    maxent_model <- maxent(x = train_data, p = train_response)
    
    # Predict on the test data
    maxent_pred <- predict(maxent_model, test_data, type = "cloglog")
    
    # Evaluate AUC
    auc_maxent <- roc(response = test_response, predictor = maxent_pred)$auc
    aucs <- c(aucs, auc_maxent)
    
    # Print AUC of each iteration
    print(paste("Iteration", i, "AUC:", auc_maxent))
    
    # Predict on the full environmental space
    maxent_raster <- terra::predict(env_layers, maxent_model, fun = function(model, data) {
      as.numeric(dismo::predict(model, data, type = "cloglog"))
    }, na.rm = TRUE)
    output_filename <- paste0("maxent_prediction_", i, ".tif")
    writeRaster(maxent_raster, filename = output_filename, overwrite = TRUE)
    
    # Store the results
    rasters[["Maxent"]] <- append(rasters[["Maxent"]], list(maxent_raster))
    model_results[["Maxent"]] <- append(model_results[["Maxent"]], auc_maxent)
  }, error = function(e) {
    print(paste("Iteration", i, "Error occurred:", e$message))
  })
}

# Calculate the average AUC
average_maxent_auc <- mean(aucs, na.rm = TRUE)
print(paste("Average Maxent AUC:", average_maxent_auc))

#----------------------------------------Model Combination/ensembling--------------------------------------
# Placeholder for model results (AUCs from previous runs)
models <- list(
  xgboost = xgb_model,
  rf = rf_model,
  maxent = maxent_model
)

# Initialize lists to store the mean AUCs and raster outputs for each model
model_results <- list(
  XGBoost = average_xgb_auc,  # Replace with the actual list of AUCs for XGBoost
  RF = average_rf_auc,        # Replace with the actual list of AUCs for RF
  Maxent = average_maxent_auc # Replace with the actual list of AUCs for Maxent
)

rasters <- list(
  XGBoost = xgb_raster, # Replace with the actual list of rasters from XGBoost
  RF = rf_raster,       # Replace with the actual list of rasters from RF
  Maxent = maxent_raster # Replace with the actual list of rasters from Maxent
)

# Calculate mean AUC for each model
mean_aucs <- sapply(model_results, function(aucs) mean(unlist(aucs), na.rm = TRUE))

# Print mean AUCs for each model
print("Mean AUCs for each model:")
print(mean_aucs)

# Combine models with mean AUC > 0.80 using a weighted mean
model_weights <- sapply(mean_aucs, function(auc) ifelse(auc > 0.80, auc, 0))
weight_sum <- sum(model_weights)

if (weight_sum == 0) {
  stop("No models with mean AUC > 0.80")
}

# Calculate the weighted sum of predictions (final combined raster)
combined_raster <- NULL
for (model_name in names(model_weights)) {
  if (model_weights[[model_name]] > 0) {
    # Average the prediction rasters for the model
    model_raster <- Reduce("+", rasters[[model_name]]) / length(rasters[[model_name]])
    
    # Weight the raster by its model's AUC and add to the final combined raster
    weighted_raster <- model_raster * (model_weights[[model_name]] / weight_sum)
    
    if (is.null(combined_raster)) {
      combined_raster <- weighted_raster
    } else {
      combined_raster <- combined_raster + weighted_raster
    }
  }
}

# Save the combined weighted raster to a file
writeRaster(combined_raster, filename = "combined_model_Biomphalaria.tif", overwrite = TRUE)
# plot(combined_raster)

# Print final results
cat("\nModel weights (based on mean AUCs):\n")
print(model_weights)
cat("\nCombined model prediction saved as 'combined_model_Biomphalaria.tif'\n")

#---------------------------------------Crossvalidation variable importance---------------------------------------
# Function for cross-validation of variable importance
cross_validate_variable_importance <- function(num_iterations = 10, model_weights, xgboost_model, rf_model, maxent_model) {
  
  # Placeholder for storing variable importance across iterations
  combined_var_imp_all_iterations <- list()
  
  for (iter in 1:num_iterations) {
    cat("Iteration:", iter, "\n")
    
    # Extract Variable Importance for Each Model in this iteration
    # XGBoost variable importance (normalized to percentage)
    xgboost_var_imp <- if ("XGBoost" %in% names(model_weights) && model_weights["XGBoost"] > 0) {
      var_imp <- xgb.importance(model = xgb_model)
      var_imp$Gain <- var_imp$Gain / sum(var_imp$Gain) * 100  # Normalize to sum to 100%
      data.frame(variable = var_imp$Feature, Contribution = var_imp$Gain)
    } else {
      NULL
    }
    
    # RF variable importance (normalized to percentage)
    rf_var_imp <- if ("RF" %in% names(model_weights) && model_weights["RF"] > 0) {
      var_imp <- data.frame(variable = rownames(importance(rf_model)), Contribution = importance(rf_model)[, "MeanDecreaseGini"])
      var_imp$Contribution <- var_imp$Contribution / sum(var_imp$Contribution) * 100  # Normalize to 100%
      var_imp
    } else {
      NULL
    }
    
    # Maxent variable importance (normalized to percentage)
    maxent_var_imp <- if ("Maxent" %in% names(model_weights) && model_weights["Maxent"] > 0) {
      var_imp <- data.frame(variable = rownames(maxent_model@results[grep("Contribution", rownames(maxent_model@results))]), 
                            Contribution = as.numeric(maxent_model@results[grep("Contribution", rownames(maxent_model@results))]))
      if (nrow(var_imp) > 0) {
        var_imp$Contribution <- var_imp$Contribution / sum(var_imp$Contribution) * 100  # Normalize to 100%
        var_imp
      } else {
        NULL
      }
    } else {
      NULL
    }
    
    #Combine and Weight Variable Importances for this iteration
    combined_var_imp <- data.frame(variable = character(), Weighted_Contribution = numeric())
    
    normalize_and_weight <- function(var_imp, model_weight) {
      var_imp$Weighted_Contribution <- var_imp$Contribution * (model_weight / sum(model_weights))  # Apply model weight
      return(var_imp[, c("variable", "Weighted_Contribution")])
    }
    
    # Combine the variable importance from each model
    if (!is.null(xgboost_var_imp)) {
      xgboost_var_imp <- normalize_and_weight(xgboost_var_imp, model_weights["XGBoost"])
      combined_var_imp <- rbind(combined_var_imp, xgboost_var_imp)
    }
    
    if (!is.null(rf_var_imp)) {
      rf_var_imp <- normalize_and_weight(rf_var_imp, model_weights["RF"])
      combined_var_imp <- rbind(combined_var_imp, rf_var_imp)
    }
    
    if (!is.null(maxent_var_imp)) {
      maxent_var_imp <- normalize_and_weight(maxent_var_imp, model_weights["Maxent"])
      combined_var_imp <- rbind(combined_var_imp, maxent_var_imp)
    }
    
    # Aggregate contributions by variable and calculate total percentage contribution
    final_var_contributions_iter <- aggregate(Weighted_Contribution ~ variable, data = combined_var_imp, sum)
    final_var_contributions_iter$Weighted_Contribution <- final_var_contributions_iter$Weighted_Contribution / 
      sum(final_var_contributions_iter$Weighted_Contribution) * 100
    
    # Store the results for this iteration
    combined_var_imp_all_iterations[[iter]] <- final_var_contributions_iter
  }
  
  # Combine results across all iterations
  combined_all <- do.call(rbind, combined_var_imp_all_iterations)
  final_var_contributions_all <- aggregate(Weighted_Contribution ~ variable, data = combined_all, mean)
  
  # Normalize and sort by contribution
  final_var_contributions_all$Weighted_Contribution <- final_var_contributions_all$Weighted_Contribution / 
    sum(final_var_contributions_all$Weighted_Contribution) * 100
  final_var_contributions_all <- final_var_contributions_all[order(-final_var_contributions_all$Weighted_Contribution), ]
  
  # Return final results
  return(final_var_contributions_all)
}

#Printing visualization variables_importance
# Print and plot results
print(final_var_contributions_all)
# Visualizing variable imprortance bar graph using ggplot
# Change labels before Plotting 
custom_labels <- c(
  "bio1" = "Annual Mean Temperature (BIO1)",
  "bio2" = "Mean Diurnal Range (BIO2)",
  "bio3" = "Isothermality (BIO3)",
  "bio4" = "Temperature Seasonality (BIO4)",
  "bio5" = "Max Temperature of Warmest Month (BIO5)",
  "bio6" = "Min Temperature of Coldest Month (BIO6)",
  "bio7" = "Temperature Annual Range (BIO7)",
  "bio8" = "Mean Temperature of Wettest Quarter (BIO8)",
  "bio9" = "Mean Temperature of Driest Quarter (BIO9)",
  "bio10" = "Mean Temperature of Warmest Quarter (BIO10)",
  "bio11" = "Mean Temperature of Coldest Quarter (BIO11)",
  "bio12" = "Annual Precipitation (BIO12)",
  "bio13" = "Precipitation of Wettest Month (BIO13)",
  "bio14" = "Precipitation of Driest Month (BIO14)",
  "bio15" = "Precipitation Seasonality (BIO15)",
  "bio16" = "Precipitation of Wettest Quarter (BIO16)",
  "bio17" = "Precipitation of Driest Quarter (BIO17)",
  "bio18" = "Precipitation of Warmest Quarter (BIO18)",
  "bio19" = "Precipitation of Coldest Quarter (BIO19)",
  "DEM"= "Elevation",
  "NDVI" = "Normalized Vegetation Index (NDVI)",
  "DW" = "Distance to Water Body",
  "Slope" = "Slope",
  "clay" = "Clay Content",
  "silt" = "Silt Content"
)
ggplot(final_var_contributions_all, aes(x = reorder(variable, Weighted_Contribution), y = Weighted_Contribution, fill = Weighted_Contribution)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Weighted_Contribution, 1)),  # Add text labels with rounded values
            hjust = 2,  # Adjust horizontal position of the text
            size = 3.5) +  # Adjust text size
  coord_flip() +
  scale_fill_gradient(low = "pink", high = "darkred") +  # Gradient color
  scale_y_continuous(expand = c(0, 0)) +  # Remove space between bars and axis
  scale_x_discrete(labels = custom_labels) +  # Relabel variables
  labs(title = "Variable Contribution to Weighted Mean Suitability Model for Biomphalaria spp.",
       x = "Variable", y = "Weighted Contribution (%)") +
  theme_bw() +
  theme(legend.position = "none")

#-------------------------------------------Generate Response Curves--------------------------------------------
# Initialize a placeholder for all response curves data
all_response_data <- data.frame()

# Extract sample values from the raster stack
env_data <- as.data.frame(terra::spatSample(env_layers, size = 1000, method = "random", xy = FALSE, na.rm = TRUE))

# Calculate the median for each environmental variable to keep them constant
constant_values <- apply(env_data, 2, median, na.rm = TRUE)

# Iterate over each environmental variable
for (var in names(env_data)) {
  cat("Generating synthetic data for variable:", var, "\n")
  
  # Generate a range of values for the selected variable
  var_range <- seq(min(env_data[[var]], na.rm = TRUE), max(env_data[[var]], na.rm = TRUE), length.out = 100)
  
  # Create synthetic data, keeping other variables constant at their median
  synthetic_data <- data.frame(matrix(rep(constant_values, each = length(var_range)), nrow = length(var_range)))
  colnames(synthetic_data) <- names(constant_values)
  synthetic_data[[var]] <- var_range  # Vary only the current variable
  
  # # Print the first few rows of the synthetic data for inspection
  cat("Synthetic data preview for", var, ":\n")
  print(head(synthetic_data))
  cat("\n")  # Print a newline for better readability
  
  # Convert the synthetic data to a matrix for xgboost prediction
  synthetic_data_matrix <- as.matrix(synthetic_data)
  
  # Generate model predictions and calculate the weighted mean
  combined_pred <- tryCatch({
    # Predict with XGBoost model, ensure synthetic data is passed correctly
    xgb_pred <- predict(xgb_model, newdata = synthetic_data_matrix)
    
    # Other model predictions
    rf_pred <- as.numeric(predict(rf_model, newdata = synthetic_data, type = "prob")[, 2])
    maxent_pred <- as.numeric(dismo::predict(maxent_model, synthetic_data))
    
    # Combine predictions using model weights
    (xgb_pred * model_weights["XGBoost"] +
        rf_pred * model_weights["RF"] +
        maxent_pred * model_weights["Maxent"]
    ) / sum(model_weights)
  }, error = function(e) {
    cat("Error in prediction for", var, ":", e$message, "\n")
    rep(NA, length(var_range))
  })
  
  # Create a data frame for the response curve of the current variable
  response_data <- data.frame(
    Variable = var,
    Value = var_range,
    Predicted_Probability = combined_pred
  )
  
  # Append to the all_response_data data frame
  all_response_data <- rbind(all_response_data, response_data)
}

# Print the first few rows of the combined response data to ensure it's created correctly
print(head(all_response_data))

# Plotting all response curves on individual plots with smooth dark red lines and free x and y scales
ggplot(all_response_data, aes(x = Value, y = Predicted_Probability)) +
  geom_line(size = 1, color = "blue") +  # Smooth dark red line
  facet_wrap(~ Variable, scales = "free") +  # Free both x and y axes for better visibility
  labs(
    x = "Environmental Variable Value",
    y = "Predicted Probability",
    title = "Response Curves of Prediction Biomphalaria spp. vs Environmental Variables"
  ) +
  theme_bw() +
  theme(
    strip.text = element_text(size = 10),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 10)
  )
#------------------------------Model Weight Computation for Future Predictions-------------------------------
# Define model weights based on AUCs (ensure you have these AUC values from training)
xgb_auc <- 0.935  # Replace with actual AUC of XGBoost
rf_auc <- 0.957   # Replace with actual AUC of Random Forest
maxent_auc <- 0.950  # Replace with actual AUC of Maxent

# Calculate weights
total_auc <- xgb_auc + rf_auc + maxent_auc
xgb_weight <- xgb_auc / total_auc
rf_weight <- rf_auc / total_auc
maxent_weight <- maxent_auc / total_auc

cat("Weights: XGBoost =", xgb_weight, ", RF =", rf_weight, ", Maxent =", maxent_weight, "\n")

#################################### Future Prediction 2041-2060 SSP 126 #####################################
#Load environmental variables of future climate scenario SSP1-2.6
env_variables_Future_126_path <- "F:/EVS_126"
# Load all the raster files
env_files126 <- list.files(env_variables_Future_126_path, pattern = "\\.tif$", full.names = TRUE)

#################################### Align and resample environmental raster files###########################
# Load the first raster as a reference for extent and resolution
ref_raster126 <- rast(env_files126[1])
# Function to check extents and reproject if necessary
check_and_align_raster_126 <- function(raster, ref_raster126) {
  # Check if CRS matches
  if (!same.crs(raster, ref_raster126)) {
    raster <- project(raster, crs(ref_raster126))
    cat("Reprojected raster to match CRS of the reference raster.\n")
  }
  
  # Check if extents overlap using `intersect()` for extents
  if (is.null(intersect(ext(raster), ext(ref_raster126)))) {
    stop("Raster extents do not overlap.")
  }
  
  # Crop and resample the raster to match the reference raster
  raster_cropped126 <- crop(raster, ext(ref_raster126))
  raster_resampled126 <- terra::resample(raster_cropped126, ref_raster126, method = "bilinear")
  
  return(raster_resampled126)
}

# Apply the function to each raster file
aligned_rasters126 <- lapply(env_files126, function(file) {
  raster <- rast(file)
  check_and_align_raster(raster, ref_raster126)
})

# Stack the aligned rasters
env_layers126 <- rast(aligned_rasters126)

# Check the final stack
print(env_layers126)
summary(env_layers126)

###################################### Weighted Ensemble Future 126 Prediction #################################
# Predict suitability using XGBoost
cat("Predicting suitability using XGBoost...\n")
xgb_suitability126 <- terra::predict(
  env_layers126,
  xgb_model,
  fun = function(model, d) {
    predict(model, xgboost::xgb.DMatrix(as.matrix(d)))
  },
  na.rm = TRUE
)

# Predict suitability using Random Forest
cat("Predicting suitability using Random Forest...\n")
rf_suitability126 <- terra::predict(
  env_layers126,
  rf_model,
  fun = function(model, d) predict(model, d, type = "prob")[, 2],
  na.rm = TRUE
)

# Predict suitability using Maxent
cat("Predicting suitability using Maxent...\n")
maxent_suitability126 <- terra::predict(env_layers126, maxent_model, fun = function(model, data) {
  as.numeric(dismo::predict(model, data, type = "cloglog"))
}, na.rm = TRUE)

# Combine predictions using weighted ensemble
cat("Combining suitability predictions using weighted ensemble...\n")
ensemble_suitability126 <- (xgb_suitability126 * xgb_weight) +
  (rf_suitability126 * rf_weight) +
  (maxent_suitability126 * maxent_weight)

# Save the final ensemble suitability map
output_file <- "Biomphalaria_future_SSP126.tif"
writeRaster(ensemble_suitability126, filename = output_file, overwrite = TRUE)
cat("Final suitability map saved to:", output_file, "\n")

# Plot the ensemble suitability
plot(ensemble_suitability126, main = "Future Habitat Suitability (SSP 126)")

#----------------------------------------Combined Future 585 Weighted Model--------------------------------

##############################  Future Prediction 2041-2060 SSP 585 ########################################
#Load environmental variables of future climate scenario SSP1-2.6
env_variables_Future_585_path <- "F:/EVS_585"
# Load all the raster files
env_files585 <- list.files(env_variables_Future_585_path, pattern = "\\.tif$", full.names = TRUE)

###################Align and resample environmental raster files######################
# Load the first raster as a reference for extent and resolution
ref_raster585 <- rast(env_files585[1])
# Function to check extents and reproject if necessary
check_and_align_raster_585 <- function(raster, ref_raster585) {
  # Check if CRS matches
  if (!same.crs(raster, ref_raster585)) {
    raster <- project(raster, crs(ref_raster585))
    cat("Reprojected raster to match CRS of the reference raster.\n")
  }
  
  # Check if extents overlap using `intersect()` for extents
  if (is.null(intersect(ext(raster), ext(ref_raster585)))) {
    stop("Raster extents do not overlap.")
  }
  
  # Crop and resample the raster to match the reference raster
  raster_cropped585 <- crop(raster, ext(ref_raster585))
  raster_resampled585 <- terra::resample(raster_cropped585, ref_raster585, method = "bilinear")
  
  return(raster_resampled585)
}

# Apply the function to each raster file
aligned_rasters585 <- lapply(env_files585, function(file) {
  raster <- rast(file)
  check_and_align_raster(raster, ref_raster585)
})

# Stack the aligned rasters
env_layers585 <- rast(aligned_rasters585)

# # Change the NDVI Name to ensure the name is NDVI
print(names(env_layers585))
names(env_layers585)[names(env_layers585) == "lyr1"] <- "NDVI"
print(names(env_layers585))

# Check the final stack
print(names(env_layers585))
print(env_layers585)
summary(env_layers585)

################################# Weighted Ensemble Future SSP 585 Prediction ####################################
# Predict suitability using XGBoost
cat("Predicting suitability using XGBoost...\n")
xgb_suitability585 <- terra::predict(
  env_layers585,
  xgb_model,
  fun = function(model, d) {
    predict(model, xgboost::xgb.DMatrix(as.matrix(d)))
  },
  na.rm = TRUE
)

# Predict suitability using Random Forest
cat("Predicting suitability using Random Forest...\n")
rf_suitability585 <- terra::predict(
  env_layers585,
  rf_model,
  fun = function(model, d) predict(model, d, type = "prob")[, 2],
  na.rm = TRUE
)

# Predict suitability using Maxent
cat("Predicting suitability using Maxent...\n")
maxent_suitability585 <- terra::predict(env_layers585, maxent_model, fun = function(model, data) {
  as.numeric(dismo::predict(model, data, type = "cloglog"))
}, na.rm = TRUE)

# Combine predictions using weighted ensemble
cat("Combining suitability predictions using weighted ensemble...\n")
ensemble_suitability585 <- (xgb_suitability585 * xgb_weight) +
  (rf_suitability585 * rf_weight) +
  (maxent_suitability585 * maxent_weight)

# Save the final ensemble suitability map
output_file <- "Biomphalaria_future_SSP585.tif"
writeRaster(ensemble_suitability585, filename = output_file, overwrite = TRUE)
cat("Final suitability map saved to:", output_file, "\n")

# Plot the ensemble suitability
plot(ensemble_suitability585, main = "Future Habitat Suitability (SSP 585)")
##############################################################################################################

# ______________________________________________________END_____________________________________________________



