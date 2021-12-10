import shapely
import geopandas
from LSM_common import (replace_valid_data, open_memory_dataset, RasterLayer, RasterStack, RasterMeta,
                        extract_by_mask)
import numpy as np
import pytest
from rasterio.crs import CRS

@pytest.fixture
def meta():
    return RasterMeta.from_bounds(height=4, width=4, count=1, crs=CRS.from_epsg(4326), left=0, bottom=-4, right=4,
                                  top=0, driver='AAIGrid', dtype='float32')

def test_geom_comparisons():
    poly1 = shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly2 = shapely.geometry.Polygon([(0, 0.1), (2, 0), (2, 2), (0, 2)])

    bool1 = poly1.equals_exact(poly2, 0.0) # should be false tolerance is too high
    bool2 = poly1.equals_exact(poly2, 0.1) #should be true tolerance is adequate

    assert(bool1==False)
    assert(bool2==True)


def test_intersection_with_holes():
    points = [(4.0, 0.75), (4, 0), (2, 0), (2, 0.75), (2.5, 0.75), (2.5, 0.6), (2.25, 0.6),\
              (2.25, 0.2), (3.25, 0.2), (3.25, 0.5), (3.5, 0.5), (3.5, 0.75), (4, 0.75)]

    test_poly = shapely.geometry.Polygon(points)

    exterior = [(0, 0), (4, 0), (4, 2), (0, 2)]
    interior = [(2.5, 1.5), (3.5, 1.5), (3.5, 0.5), (2.5, 0.5)]

    poly1 = shapely.geometry.Polygon(exterior, [interior])
    poly2 = shapely.geometry.Polygon([(6, 0), (8, 0), (8, 1), (6, 1)])

    df1 = geopandas.GeoDataFrame({'geometry': [poly1, poly2], 'df1': [1, 2]})

    exterior = [(2, 0), (4, 0), (4, 0.75), (2, 0.75)]
    interior = [(2.25, 0.6), (3.25, 0.6), (3.25, 0.2), (2.25, 0.2)]
    poly1 = shapely.geometry.Polygon(exterior, [interior])

    df2 = geopandas.GeoDataFrame({'geometry': [poly1], 'df2': [1]})

    res_intersection = geopandas.overlay(df1, df2, how='intersection')

    assert (test_poly.equals_exact(res_intersection.geometry.values[0], 1E-16))

def test_replace_valid_data_1():
    data = np.array([[1, 3, 8, np.nan], [np.nan, 6, 7, 9], [np.nan, 0, 1, 2], [5, np.nan, np.nan, 2]])
    array = replace_valid_data(data, 2)
    test_array = np.array([[2, 2, 2, np.nan], [np.nan, 2, 2, 2], [np.nan, 2, 2, 2], [2, np.nan, np.nan, 2]]).astype(data.dtype)
    np.testing.assert_equal(test_array, array)



def test_generate_valid_data_array_2(meta):
    r1 = RasterLayer(r'test_input/test_generate_valid_data_array_2_1.asc', 'r1')
    r2 = RasterLayer(r'test_input/test_generate_valid_data_array_2_2.asc', 'r2')
    stack = RasterStack([r1, r2], meta)

    valid_data_array = stack.generate_valid_data_array()
    stack.close()
    test_valid_data_array = np.array([[np.nan, 1, 1, np.nan], [np.nan, 1, 1, 1], [np.nan, np.nan, 1, 1],
                                      [1, np.nan, np.nan, 1]]).astype(valid_data_array.dtype)

    np.testing.assert_equal(valid_data_array, test_valid_data_array)

def test_generate_valid_data_polygon_2(meta):
    r1 = RasterLayer(r'test_input/test_generate_valid_data_polygon_2_1.asc', 'r1')
    r2 = RasterLayer(r'test_input/test_generate_valid_data_polygon_2_2.asc', 'r2')
    stack = RasterStack([r1, r2], meta)

    gdf, array = stack.generate_valid_data_polygon()
    stack.close()

    poly1 = shapely.geometry.Polygon([(1, 0), (1, -2), (2, -2), (2, -3), (3, -3),
                                      (3, -4), (4, -4), (4, -1), (3, -1), (3, 0), (1, 0)])
    poly2 = shapely.geometry.Polygon([(0, -3), (0, -4), (1, -4), (1, -3), (0, -3)])

    assert poly1.equals_exact(gdf.geometry[0], 1E-16)
    assert poly2.equals_exact(gdf.geometry[1], 1E-16)


def test_extract_by_mask_1():
    #check if polygon extracts correct cells by checking if the sum of the cell values is correct (diagonal polygon)
    r_data = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]).astype('float32')
    r = open_memory_dataset(r_data, 0, 2, 0.5)

    p = shapely.geometry.Polygon([[0, 0], [0.3, 0.1], [1.58, 1.5], [2, 2], [1.56, 1.8], [0.1, 0.4]])
    m = extract_by_mask(r, [p])

    r.close()

    np.testing.assert_equal(np.array([[np.nan, np.nan, np.nan, 4], [np.nan, np.nan, 7, np.nan],
                                      [np.nan, 10, np.nan, np.nan], [13, np.nan, np.nan, np.nan]]), m)

def test_extract_by_mask_2():
    #check if polygon extracts correct cells by checking if the sum of the cell values is correct (diagonal polygon)
    r_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]).astype('float32')
    r = open_memory_dataset(r_data, 0, 2, 0.5)
    p = shapely.geometry.Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    m = extract_by_mask(r, [p])

    r.close()

    np.testing.assert_equal(m, np.array([4*[np.nan],4*[np.nan],[9,10, np.nan, np.nan],[13,14, np.nan, np.nan]]))


def test_extract_by_mask_3():
    #extract by mask using linestring object instead of polygon and all_touched=True
    r_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]).astype('float32')
    r = open_memory_dataset(r_data, 0, 2, 0.5)
    linestring = shapely.geometry.LineString([(0.25, 0), (2, 1.75)])
    m = extract_by_mask(r, [linestring], True) # here the all_touched parameter was introduced as an optional argument.

    r.close()
    np.testing.assert_equal(m, np.array([[np.nan, np.nan, np.nan, 4],[np.nan, np.nan, 7,8],
                                           [np.nan,10,11,np.nan],[13,14,np.nan, np.nan]]))

def test_extract_by_mask_4():
    # check if polygon extracts correct cells with inversion (outside polygon)
    r_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]).astype('float32')

    r = open_memory_dataset(r_data, 0, 2, 0.5)

    p = shapely.geometry.Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    m = extract_by_mask(r, [p], invert=True)
    r.close()

    np.testing.assert_equal(m, np.array([[1, 2, 3, 4],[5, 6, 7, 8],
                                       [np.nan, np.nan, 11, 12], [np.nan, np.nan,15, 16]]))

