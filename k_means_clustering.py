import numpy as np
from sklearn import cluster
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
from PIL import Image
import tarfile
import os

def kmeans(input_path: str, num_clusters: int, output_path: str):
    gdal.UseExceptions()
    gdal.AllRegister()
    img_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    num_rasters = img_ds.RasterCount
    
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, num_rasters),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

    for b in range(num_rasters):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    
    X = img.reshape((-1, num_rasters))
    
    k_means = cluster.KMeans(n_clusters=num_clusters)
    k_means.fit(X)
    
    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img.shape[:2])

    [cols, rows] = img_ds.GetRasterBand(1).ReadAsArray().shape
    
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    
    out_data_raster = driver.Create(output_path, rows, cols, 1, gdal.GDT_Byte)
    out_data_raster.SetGeoTransform(img_ds.GetGeoTransform())
    out_data_raster.SetProjection(img_ds.GetProjection())
    
    out_data_raster.GetRasterBand(1).WriteArray(X_cluster)
    out_data_raster.FlushCache() 
    del out_data_raster

    return X_cluster

