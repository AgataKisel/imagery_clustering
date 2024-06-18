import numpy as np
from sklearn.cluster import KMeans, OPTICS
from osgeo import gdal, gdal_array 
import os

def kmeans(input_path: str, num_clusters: int=2, init: str="k-means++", max_iter: int=300, algorithm: str="lloyd", output_path: str) -> int:
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
        1 if invalid data source
        2 if error in clustering
        3 if error in file creation
        4 if invalid parameters
    Description 
    -----------
    based on https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    if not init in ["k-means++", "random"]:
        return 4
    if not algorithm in ["lloyd", "elkan"]:
        return 4
    if not isinstance(num_clusters, int):
        return 4
    if not isinstance(max_iter, int):
        return 4
    gdal.UseExceptions()
    gdal.AllRegister()
    img_ds = gdal.Open(input_path, gdal.GA_ReadOnly)

    if (img_ds==None):
        return 1 
    
    num_rasters = img_ds.RasterCount

    img = img_ds.ReadAsArray()
    img = np.moveaxis(img, 0, -1)
    
    X = img.reshape((-1, num_rasters))
    
    try:
        model = KMeans(n_clusters=num_clusters)
        X_cluster = model.fit_predict(X)
        X_cluster = X_cluster.reshape(img.shape[:2])
    except: 
        return 2
    
    format = "GTiff"
    driver = gdal.GetDriverByName(format)

    try:
        out_data_raster = driver.Create(output_path, img_ds.RasterXSize, img_ds.RasterYSize, 1, gdal.GDT_Byte)
        out_data_raster.SetGeoTransform(img_ds.GetGeoTransform())
        out_data_raster.SetProjection(img_ds.GetProjection())
    
        out_data_raster.GetRasterBand(1).WriteArray(X_cluster)
        out_data_raster.FlushCache() 
        del out_data_raster
    except:
        return 3

    return 0

