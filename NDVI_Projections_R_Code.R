#R Code for Projection of NDVI under future climate change scenarios SSP1-2.6 and SSP5-8.5 

# Project: Climate change drives small scale changes in snail habitat suitability and Schistosomiasis Risk in the Lake Victoria Basin in Sub-Saharan Africa


#-------------------------------------------------Load necessary libraries-------------------------------------------
# Load necessary libraries
library(terra)        # For raster manipulation
library(caret)        # For cross-validation and performance metrics
library(randomForest) # For Random Forest regression
library(ggplot2)      # For visualizations
library(sf)           # For shapefile processing
library(tidyverse)
#----------------------------------Load required region of interest file and data sets -------------------------------- 
# Load region of interest (ROI) shapefile
roi <- st_read("C:/Users/DELL/Documents/NDVI_Prediction/LVB_extent/LVB_extent.shp")

# Load NDVI raster (current year, e.g., 2023)
ndvi <- rast("C:/Users/DELL/Documents/NDVI_Prediction/Other_Var/ndvi_resampled.tif")

# Load topographic variables
elevation <- rast("C:/Users/DELL/Documents/NDVI_Prediction/Other_Var/elevation_resampled.tif")
slope<- rast("C:/Users/DELL/Documents/NDVI_Prediction/Other_Var/slope_resampled.tif")
plot(elevation)

# Load all bioclimatic variables from the folder
bioclim_folder <- "C:/Users/DELL/Documents/R/Bioclim"
bioclim_stack <- rast(list.files(bioclim_folder, pattern = "\\.tif$", full.names = TRUE))

#---------------------------------- Process and align all data sets to ensure uniformity ----------------------------
# clip the bioclim and other varaibales to ROI
bioclim_stack_clip <- crop(bioclim_stack, roi)
elevation_clipped <- crop(elevation, roi)
slope_clipped <- crop(slope, roi)

# Ensure all rasters have the same CRS
target_crs <- crs(ndvi)

# Reproject rasters to the target CRS (NDVI CRS)
bioclim_rep <- project(bioclim_stack_clip, target_crs)
elevation_rep <- project(elevation_clipped, target_crs)
slope_rep <- project(slope_clipped, target_crs)

# Resample all rasters to the NDVI raster's resolution and extent
bioclim_resampled <- resample(bioclim_rep, ndvi)  # Reference NDVI for resampling
elevation_resampled <- resample(elevation_rep, ndvi)
slope_resampled <- resample(slope_rep, ndvi)

# Check individual raster summaries
summary(ndvi)
summary(bioclim_resampled)
summary(elevation_resampled)
summary(slope_resampled)

# Verify CRS alignment
cat("NDVI CRS:", crs(ndvi), "\n")
cat("Bioclim CRS:", crs(bioclim_resampled), "\n")
cat("Elevation CRS:", crs(elevation_resampled), "\n")
cat("Slope CRS:", crs(slope_resampled), "\n")

# Optional: Check geometry alignment
if (compareGeom(ndvi, bioclim_resampled) &&
    compareGeom(ndvi, elevation_resampled) &&
    compareGeom(ndvi, slope_resampled)) {
  cat("All rasters are aligned.\n")
} else {
  cat("Geometry misalignment detected.\n")
}

# Combine variables into a stack for sample points extraction
combined_vars <- c(elevation_resampled, slope_resampled, bioclim_resampled, ndvi)

#--------------------------------------Generate random sample points from the study area-------------------------------
set.seed(100)
sample_points <- spatSample(combined_vars, size = 5000, as.points = TRUE, method = "random")
plot(sample_points)

# Convert to a data frame
sample_points_df <- as.data.frame(sample_points)
print(sample_points_df)

# Rename column ndvi_2023 to NDVI  
sample_points_df <- sample_points_df %>%
  rename(NDVI = ndvi_2023)
print(sample_points_df)

# Remove rows with missing values
sample_points_clean <- sample_points_df[complete.cases(sample_points_df), ]

# check if NAs have been removed from the sampling points data set 
summary(sample_points_clean)

#---------------------------------Random forest Model training and Analysis------------------------------------------------------------
# Combine predictor variables into stack for model training 
predictors <- c(elevation_resampled, slope_resampled, bioclim_resampled)
summary(predictors)

# Cross-validation setup
train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Train the Random Forest model with cross-validation
set.seed(100)                               # Ensure reproducibility
rf_cv_model <- train(
  NDVI ~ .,                                 # Use all predictors
  data = sample_points_clean,               # Cleaned data
  method = "rf",                            # Random Forest method
  trControl = train_control,                # Cross-validation control
  tuneLength = 5,                           # Test 10 different mtry values
  ntree = 500,                              # Specify 500 trees
  importance = TRUE                         # Measure variable importance
)

# Print cross-validation results
print(rf_cv_model)

# Summary of cross-validation metrics
cat("Cross-Validation Metrics:\n")
cat("RMSE:", mean(rf_cv_model$results$RMSE), "\n")
cat("R²:", mean(rf_cv_model$results$Rsquared), "\n")

# Plot variable importance
varImpPlot(rf_cv_model$finalModel, main = "Variable Importance (Cross-Validation)")

# Predict NDVI across the entire study area (2023)
predicted_ndvi_2023 <- predict(predictors, rf_cv_model$finalModel)
plot(predicted_ndvi_2023)
plot(ndvi)

# Save predicted NDVI raster for 2023
writeRaster(predicted_ndvi_2023, "NDVI_2023.tif", overwrite = TRUE)

############################### Future Projection of NDVI under SSP126 #############################################

# Load all future SSP126 bioclimatic variables from the folder
SSP126_bioclim <- "D:/Climate/SSP_1_2041_2060"
SSP126_bioclim_stack <- rast(list.files(SSP126_bioclim, pattern = "\\.tif$", full.names = TRUE))

# Verify the current layer names
original_names <- names(SSP126_bioclim_stack)
print("Original Names:")
print(original_names)


# Extract the numerical suffix from layer names (e.g., "_1", "_2", ..., "_19")
suffixes <- as.numeric(sub(".*_(\\d+)$", "\\1", original_names))

# Reorder the layers based on the numerical suffix
ordered_stack <- SSP126_bioclim_stack[[order(suffixes)]]
print(ordered_stack)
# Rename the layers to bio1, bio2, ..., bio19
new_names <- paste0("bio", 1:19)
names(ordered_stack) <- new_names

# Verify the updated layer names
print("Reordered and Renamed Layers:")
print(names(ordered_stack))

# Check the new layer names
print(names(ordered_stack))

# Plot one of the layers to verify
plot(ordered_stack[[1]], main = "Bio1")

# clip the future bioclim to ROI
bioclim_stack_SSP126 <- crop(ordered_stack, roi)

# Reproject rasters to the target CRS (NDVI CRS)
bioclim_rep_SSP126 <- project(bioclim_stack_SSP126, target_crs)

# Resample all rasters to the NDVI raster's resolution and extent
bioclim_resamp_SSP126 <- resample(bioclim_rep_SSP126, ndvi)  # Reference NDVI for resampling

# Combine predictor variables into stack for model training 
predictors_SSP126 <- c(elevation_resampled, slope_resampled, bioclim_resamp_SSP126)
summary(predictors_SSP126)

# Predict NDVI across the entire study area (2023)
predicted_ndvi_SSP126 <- predict(predictors_SSP126, rf_cv_model$finalModel)
plot(predicted_ndvi_SSP126)
plot(ndvi)

# Save predicted NDVI raster for SSP126
writeRaster(predicted_ndvi_SSP126, "NDVI_126.tif", overwrite = TRUE)


############################### Future Projection of NDVI under SSP585 #############################################

# Load all future SSP126 bioclimatic variables from the folder
SSP585_bioclim <- "D:/Climate/SSP_5_2041_2060"
SSP585_bioclim_stack <- rast(list.files(SSP585_bioclim, pattern = "\\.tif$", full.names = TRUE))

# Verify the current layer names
original_names <- names(SSP585_bioclim_stack)
print("Original Names:")
print(original_names)


# Extract the numerical suffix from layer names (e.g., "_1", "_2", ..., "_19")
suffixes <- as.numeric(sub(".*_(\\d+)$", "\\1", original_names))

# Reorder the layers based on the numerical suffix
ordered_stack <- SSP585_bioclim_stack[[order(suffixes)]]
print(ordered_stack)
# Rename the layers to bio1, bio2, ..., bio19
new_names <- paste0("bio", 1:19)
names(ordered_stack) <- new_names

# Verify the updated layer names
print("Reordered and Renamed Layers:")
print(names(ordered_stack))

# Check the new layer names
print(names(ordered_stack))

# Plot one of the layers to verify
plot(ordered_stack[[1]], main = "Bio1")

# clip the future bioclim to ROI
bioclim_stack_SSP585 <- crop(ordered_stack, roi)

# Reproject rasters to the target CRS (NDVI CRS)
bioclim_rep_SSP585 <- project(bioclim_stack_SSP585, target_crs)

# Resample all rasters to the NDVI raster's resolution and extent
bioclim_resamp_SSP585 <- resample(bioclim_rep_SSP585, ndvi)  # Reference NDVI for resampling

# Combine predictor variables into stack for model training 
predictors_SSP585 <- c(elevation_resampled, slope_resampled, bioclim_resamp_SSP585)
summary(predictors_SSP585)

# Predict NDVI across the entire study area (2023)
predicted_ndvi_SSP585 <- predict(predictors_SSP585, rf_cv_model$finalModel)
plot(predicted_ndvi_SSP585)
plot(ndvi)

# Save predicted NDVI raster for 2023
writeRaster(predicted_ndvi_SSP585, "NDVI_585.tif", overwrite = TRUE)

#####################################################################################################################

#__________________________________________________________END_______________________________________________________
