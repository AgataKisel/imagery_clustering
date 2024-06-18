from k_means_clustering import kmeans
import unittest
import os 
import numpy
from osgeo import gdal

class TestClustering(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    def test_3_band(self):
        """Testing 3_band geotif"""
        kmeans(input_path=os.path.join(self.dir_path, "data", "3_band.tif"), 
              num_clusters=3, output_path=os.path.join(self.dir_path, "tests", "3_band.tif"))
        ds = gdal.Open(os.path.join(self.dir_path, "tests", "3_band.tif"), gdal.GA_ReadOnly)
        ds_data = ds.ReadAsArray()
        unique_values = numpy.unique(ds_data)
        self.assertEqual(len(unique_values), 3)

    def test_1_band(self):
        """Testing 1_band geotif"""
        kmeans(input_path=os.path.join(self.dir_path, "data", "1_band.tif"), 
              num_clusters=5, output_path=os.path.join(self.dir_path, "tests", "1_band.tif"))
        ds = gdal.Open(os.path.join(self.dir_path, "tests", "1_band.tif"), gdal.GA_ReadOnly)
        ds_data = ds.ReadAsArray()
        unique_values = numpy.unique(ds_data)
        self.assertEqual(len(unique_values), 5)
        
    def test_rgba(self):
        """Testing rgba geotif"""
        kmeans(input_path=os.path.join(self.dir_path, "data", "rgba.tiff"), 
              num_clusters=4, output_path=os.path.join(self.dir_path, "tests", "rgba.tif"))
        ds = gdal.Open(os.path.join(self.dir_path, "tests", "rgba.tif"), gdal.GA_ReadOnly)
        ds_data = ds.ReadAsArray()
        unique_values = numpy.unique(ds_data)
        self.assertEqual(len(unique_values), 4)
        
    def test_topomap(self):
        """Testing topomap png"""
        kmeans(input_path=os.path.join(self.dir_path, "data", "topomap.png"), 
              num_clusters=2, output_path=os.path.join(self.dir_path, "tests", "topomap.tif"))
        ds = gdal.Open(os.path.join(self.dir_path, "tests", "topomap.tif"), gdal.GA_ReadOnly)
        ds_data = ds.ReadAsArray()
        unique_values = numpy.unique(ds_data)
        self.assertEqual(len(unique_values), 2)
if __name__ == '__main__':
    unittest.main()