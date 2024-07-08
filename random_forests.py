from __future__ import print_function, division
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, ogr
import os
os.environ['PROJ_LIB'] = 'C:\\Users\\AK\\anaconda3\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\AK\\anaconda3\\pkgs\\proj-9.3.1-ha107b6e_0\\Library\\share\\proj'

def load_shape_dataset(filepath, img_ds):
    """
    Parametrs
    ---------
    filepath: str
        path to source gdal-compatible traning vector

    img_ds: osgeo.gdal.Dataset
        initial image with goespatial reference
        
    Returns
    --------
    dataset: numpy.ndarray
        raster of traning  

    """
    shape_dataset = ogr.Open(filepath)
    shape_layer = shape_dataset.GetLayer()
    attributes = []
    ldefn = shape_layer.GetLayerDefn()
    
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        attributes.append(fdefn.name)  

    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16)
    mem_raster.SetProjection(img_ds.GetProjection())
    mem_raster.SetGeoTransform(img_ds.GetGeoTransform())
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(0)
    mem_band.SetNoDataValue(0)

    err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1],  ['ALL_TOUCHED=TRUE', 'ATTRIBUTE=id'])
    assert err == gdal.CE_None
    dataset = mem_raster.ReadAsArray()
    return dataset

def random_forests(input_path_initial_image: str, input_path_traning_data: str, output_path: str) -> int:
    """
    Performs random forests classification of any gdal-compatible raster
    
    Parametrs
    ---------
    input_path: str
        path to source gdal-compatible initial raster

    input_path_traning_data: str
        path to source gdal-compatible traning vector

    output_path: str
        path to create new raster in

    Returns
    --------
    int 
        0 if finished correct
        1 if invalid data source
        2 if error in classification
        3 if error in file creation
    Description 
    -----------
    based on https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    
    gdal.UseExceptions()
    gdal.AllRegister()
    
    img_ds = gdal.Open(input_path_initial_image, gdal.GA_ReadOnly)

    if (img_ds==None):
        return 1 
    
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    training = input_path_traning_data
    roi = load_shape_dataset(training, img_ds)
    num_rasters = img_ds.RasterCount
    
    X = img[roi > 0, :]
    y = roi[roi > 0]
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=True)
    
    try:
        rf = RandomForestClassifier(n_estimators=500, oob_score=True)
        rf = rf.fit(X_train, y_train)
    except: 
        return 2

    accuracy = rf.oob_score_ * 100
    print(accuracy)
    
    prediction = np.where(img[:,:,0] > 0, rf.predict(img.reshape(-1, num_rasters)).reshape(img.shape[:2]), img[:,:,0])

    try:
        format = "GTiff"
        driver = gdal.GetDriverByName(format)
        out_data_raster = driver.Create(output_path, img_ds.RasterXSize, img_ds.RasterYSize, 1, gdal.GDT_Byte)
        out_data_raster.SetGeoTransform(img_ds.GetGeoTransform())
        out_data_raster.SetProjection(img_ds.GetProjection())
        
        out_data_raster.GetRasterBand(1).WriteArray(prediction)
        out_data_raster.FlushCache() 
        del out_data_raster
    except:
        return 3
        
    return 0
    
        