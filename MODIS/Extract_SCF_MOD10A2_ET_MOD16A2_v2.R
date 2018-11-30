## Code for calculating snow cover fraction from MODIS MOD10A2 and ET based on MODIS MOD16A2 as catchment means

# Developed for "Kratzert et al. (2018) Do internals of neural networks make sense in the context of hydrology?" presented at the AGU 2018

# the user has to adapt (i) the paths to the files downloaded by MODIStsp(), (ii) the name and path to the shape-file containing the catchments and 
#  (iii) the paths to the location, where outputs should be saved

# contact: mathew.herrnegger[a]boku.ac.at; November 2018


library(gWidgetsRGtk2)
library(MODIStsp)

library(raster)
library(doParallel)
library(foreach)



### (1) Calculate snow cover fraction based on MODIS MOD10A2 ##########

# open GUI to download data
MODIStsp() # see http://ropensci.github.io/MODIStsp/
##citation: L. Busetto, L. Ranghetti (2016) MODIStsp: An R package for automatic preprocessing of MODIS Land Products time series, Computers & Geosciences, Volume 97, Pages 40-48, ISSN 0098-3004, http://dx.doi.org/10.1016/j.cageo.2016.08.020, URL: https://github.com/ropensci/MODIStsp.


# Now load a shapefile containing polygons from which we want to extract data
setwd("E:/Kratzert_et_al_AGU2018/Shape")
polygons <- rgdal::readOGR(dsn = getwd(), "671_HUCS_Shape")
shape_index <- "hru_id" # the attribute name in the attribute table of the shape-file, for which data will be aggregated
featID <- as.character(polygons$hru_id)

# function to calculate snow cover fraction
SCF_MODIS_SNOW <- function(x){
  # https://nsidc.org/data/mod10a2
  # 0: missing data
  # 1: no decision
  # 11: night
  # 25: no snow
  # 37: lake
  # 39: ocean
  # 50: cloud
  # 100: lake ice
  # 200: snow
  # 254: detector saturated
  # 255: fill
  Thresh <- 0.5
  snow <- sum(x == 200, na.rm = TRUE) # count number of snow covered pixels
  no_snow <- sum(x == 25, na.rm = TRUE) # count number of not snow covered pixels
  lake_ocean <- sum(x == 37 | x == 39 | x == 100, na.rm = TRUE) # count number of lake, ocean, lake ice pixels
  error <- sum(x == 0 | x == 1| x == 11| x == 50 | x == 254 | x == 255, na.rm = TRUE) + sum(is.na(x)) # count number of cloud or other problematic/erroneous pixels
  SCF <- snow/(snow + no_snow)
  if (error/length(x) >= Thresh) SCF <- NA # in case more pixels than the fraction defined in "Thresh" are missing data, no decision, night, detector saturated and fill, SCF is defined to be NA.
  return(SCF)
}

## extract basin values ###
pathToTifs <- "E:/Kratzert_et_al_AGU2018/MODIS/Snow_Cov_8-Day_500m_v6/MAX_SNW"
setwd(pathToTifs)

## Load raster file names
rastFiles <- list.files(path = pathToTifs, pattern = '\\.tif$')
# from first file, get crs and transform basin polygon
r <- raster(rastFiles[1])
polygons_crs_r <- spTransform(polygons, crs(r))

# extract year and doy from raster-filenames (SnowCov-Files)
years <- as.integer(substr(rastFiles, 18, 21))
doy <-  as.integer(substr(rastFiles, 23, 25))


# do the whole procedure in parallel
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cores = cl)

MaxF <- length(rastFiles)
#ptime_MHe_parallel_I <- system.time({
  Sys.time()
  # at first, extract intersecting cellnumbers of single polygons from first raster file
  c <- raster(rastFiles[1])
  beginCluster(n = parallel::detectCores() - 1)
  t <- extract(c, polygons_crs_r, cellnumbers = TRUE)
  endCluster()
  
  out <- foreach(f = 1:MaxF, .packages = c("raster", "foreach"), .combine = rbind) %dopar% {
    r_val <- values(raster(rastFiles[f]))
    final <- data.frame()
    foreach(i = 1:length(polygons_crs_r), .combine = cbind ) %dopar% {
      s <- t[[i]][, 1] # select cellnumbers for every polygon
      pol_val <- r_val[s]  
      final <- SCF_MODIS_SNOW(pol_val) #calculate mean values with function 
    }
  }
  
#})
 
  # extract date from day of year and year vectors
my_date <- strptime(paste(years, doy), format = "%Y %j") 
my_date$zone <- NULL
  
# prepare final output
SCF_basins <- as.data.frame(cbind(years[1:MaxF], doy[1:MaxF], round(out, 4)))
SCF_basins <- cbind(my_date, SCF_basins)
colnames(SCF_basins) <- c("date", "yyyy", "doy", featID)

# save results to disk
write.table(SCF_basins, "E:/Kratzert_et_al_AGU2018/MODIS/SCF_basins_MOD10A2.txt", row.names = FALSE, col.names = TRUE)



## do some plotting
plot(SCF_basins$date, SCF_basins$`1013500`, type = "l", lwd = 1, ylim = c(0,1))

k <- ncol(SCF_basins)
for(i in 5:k){
  points(SCF_basins$date, SCF_basins[,i], type = "l")
}





### (2) ET: MOD16A2 - MODIS/Terra Net Evapotranspiration 8-Day L4 Global 500 m SIN Grid V006 ############
# manual for ET: https://lpdaac.usgs.gov/sites/default/files/public/product_documentation/mod16_v6_user_guide.pdf

# open GUI to download data
MODIStsp()

# Now load a shapefile containing polygons from which we want to extract data
setwd("E:/Kratzert_et_al_AGU2018/Shape")
polygons <- rgdal::readOGR(dsn = getwd(), "671_HUCS_Shape") #9_HUCS_Shape_Test")
shape_index <- "hru_id"
featID <- as.character(polygons$hru_id)


# function to calculate mean value for intersecting grid cells, thereby removing some grid cells
meanMODIS_ET <- function(x){
  # 32767 = _Fillvalue
  # 32766 = land cover assigned as perennial salt or Water bodies
  # 32765 = land cover assigned as barren,sparse veg (rock,tundra,desert)
  # 32764 = land cover assigned as perennial snow,ice.
  # 32763 = land cover assigned as "permanent" wetlands/inundated marshland
  # 32762 = land cover assigned as urban/built-up
  # 32761 = land cover assigned as "unclassified" or (not able to determine)?
  
  x[x > 30000] <- NA # set to NA for land cover, for which ET has not been calculated
  x[x < 0] <- NA # negative values to NA (just in case)
  return((mean(x, na.rm = TRUE) * 0.1)) # multiply with 0.1 to get  mm/ 8 days sums
}

## extract basin values ###
pathToTifs <- "E:/Kratzert_et_al_AGU2018/MODIS/Net_ET_8Day_500m_v6/ET_500m"
setwd(pathToTifs)

## Load raster file names
rastFiles <- list.files(path = pathToTifs, pattern = '\\.tif$')
# from first file, get crs and transform basin polygon
r <- raster(rastFiles[1])
polygons_crs_r <- spTransform(polygons, crs(r))

# extract year and doy from raster-filenames (ET-Files)
years <- as.integer(substr(rastFiles, 17, 20))
doy <-  as.integer(substr(rastFiles, 22, 24))


## Extract basin values
MaxF <- length(rastFiles)

#ptime_MHe_parallel_I <- system.time({
Sys.time()
# at first, extract intersecting cellnumbers of single polygons from first raster file
c <- raster(rastFiles[1])
beginCluster(n = parallel::detectCores() - 1)
t <- extract(c, polygons_crs_r, cellnumbers = TRUE)
endCluster()


cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cores = cl)

out <- foreach(f = 1:MaxF, .packages = c("raster", "foreach"), .combine = rbind) %dopar% {
  r_val <- values(raster(rastFiles[f]))
  final <- data.frame()
  foreach(i = 1:length(polygons_crs_r), .combine = cbind ) %dopar% {
    s <- t[[i]][, 1] # select cellnumbers for every polygon
    pol_val <- r_val[s]  
    final <- meanMODIS_ET(pol_val) #calculate mean values with function 
  }
}

#})
#ptime_MHe_parallel_I <- system.time({
Sys.time()

# extract date from day of year and year vectors
my_date <- strptime(paste(years, doy), format="%Y %j") 
my_date$zone <- NULL

# prepare final output
ET_basins <- as.data.frame(cbind(years[1:MaxF], doy[1:MaxF], round(out, 4)))
ET_basins <- cbind(my_date, ET_basins)
colnames(ET_basins) <- c("date", "yyyy", "doy", featID)

# save results to disk
write.table(ET_basins, "E:/Kratzert_et_al_AGU2018/MODIS/ET_basins_MOD16A2.txt", row.names = FALSE, col.names = TRUE)

## do some plotting
plot(ET_basins$date, ET_basins$`1013500`/8, type = "l", lwd = 1, ylim = c(0, 6))

k <- ncol(ET_basins)
for(i in 5:k){
  points(ET_basins$date, ET_basins[,i]/8, type = "l")
}

