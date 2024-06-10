import numpy as np
from sklearn import cluster
from osgeo import gdal, gdal_array 
import os

def kmeans(input_path: str, num_clusters: int, output_path: str) -> int:
    """
    Performs k-means clustering of any gdal-compatible raster
    
    Parametrs
    ---------
    input_path: str
        path to source gdal-compatible raster

    num_clusters: int
        number of clusters to be created

    output_path: str
        path to create new raster in


    Returns
    --------
    int 
        0 if finished correct
        1 if not

    Description 
    -----------
    based on https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    
    gdal.UseExceptions()
    gdal.AllRegister()
    img_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    
    num_rasters = img_ds.RasterCount

    img = imd_ds.ReadAsArray()
    img = np.moveaxis(img, 0, -1)
    
    X = img.reshape((-1, num_rasters))
    
    k_means = cluster.KMeans(n_clusters=num_clusters)
    k_means.fit(X)
    
    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img.shape[:2])
    
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    
    out_data_raster = driver.Create(output_path, img_ds.RasterYSize, img_ds.RasterXSize, 1, gdal.GDT_Byte)
    out_data_raster.SetGeoTransform(img_ds.GetGeoTransform())
    out_data_raster.SetProjection(img_ds.GetProjection())
    
    out_data_raster.GetRasterBand(1).WriteArray(X_cluster)
    out_data_raster.FlushCache() 
    del out_data_raster

    return 0

