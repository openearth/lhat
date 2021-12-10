import pytest
import os
from pathlib import Path
import filecmp
import LSM

int_dir_test = Path('test_intermediate')
output_dir_test = Path('test_output')
int_dir_ref = Path('test_intermediate/reference')
output_dir_ref = Path('test_output/reference')

@pytest.fixture(scope="module", params = ['LR', 'SVM'])
def run_analysis(request):
    LSM.main(request.param)
    yield request.param

def test_result(run_analysis):
    test_file = output_dir_test / 'result_{}.tif'.format(run_analysis)
    ref_file = output_dir_ref / 'result_{}.tif'.format(run_analysis)
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)

def test_train_data_extracted(run_analysis):
    test_file = output_dir_test / 'train_data_extracted.geojson'
    ref_file = output_dir_ref / 'train_data_extracted.geojson'
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)

def test_buffered_random_points(run_analysis):
    test_file = int_dir_test / 'buffered_random_points.geojson'
    ref_file = int_dir_ref / 'buffered_random_points.geojson'
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)

def test_data_outside_buffered_points(run_analysis):
    test_file = int_dir_test / 'data_outside_buffered_points.tif'
    ref_file = int_dir_ref / 'data_outside_buffered_points.tif'
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)

def test_landslide_points_buffered(run_analysis):
    test_file = int_dir_test / 'landslide_points_buffered.geojson'
    ref_file = int_dir_ref / 'landslide_points_buffered.geojson'
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)

def test_train_data(run_analysis):
    test_file = int_dir_test / 'train_data.geojson'
    ref_file = int_dir_ref / 'train_data.geojson'
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)

def test_valid_data_polygon(run_analysis):
    test_file = int_dir_test / 'valid_data_polygon.geojson'
    ref_file = int_dir_ref / 'valid_data_polygon.geojson'
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)

def test_valid_data(run_analysis):
    test_file = int_dir_test / 'valid_data.tif'
    ref_file = int_dir_ref / 'valid_data.tif'
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)

def test_valid_data_unbuffered_polygon(run_analysis):
    test_file = int_dir_test / 'valid_data_unbuffered.geojson'
    ref_file = int_dir_ref / 'valid_data_unbuffered.geojson'
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)

def test_valid_data_unbuffered(run_analysis):
    test_file = int_dir_test / 'valid_data_unbuffered.tif'
    ref_file = int_dir_ref / 'valid_data_unbuffered.tif'
    assert filecmp.cmp(ref_file, test_file)
    os.remove(test_file)