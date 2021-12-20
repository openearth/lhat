import pathlib, rasterio, warnings, math, ee, urllib
import rioxarray as rx
from rasterio.enums import Resampling
from zipfile import ZipFile
import geopandas as gpd
import numpy as np
from owslib.wcs import WebCoverageService
from pyproj import Transformer
import pandas as pd
from . import Model as md
from osgeo import gdal, ogr, osr

# Initialize earth engine. If this doesn't work,
# check that you have an Earth Engine account AND
# that you entered the authentication code correctly.

#ee.Initialize()

class inputData:
    '''
    Parameters
    ----------

    name: str
        Name of input data. Eg. 'road', 'dem', 'vegetation', etc..

    path: str
        Absolute or relative path to where the input data is located

    dtype: str ('numerical' or 'categorical')
        The data type is important to define as categorical and numerical datasets
        are treated differently in the IO module.

    Returns
    -------

    :class:`inputData`:
        An object containing attributes of input data including name, filepath
        and data type
    '''
    def __init__(self, name, path, dtype):
        self.name = name
        self.path = pathlib.Path(path)
        self.dtype = dtype

class inputs:
    '''
    The inputs class initialises a project and saves all relevant data within
    the project folder.


    Initialize project with various inputs.

    Parameters
    ----------

    project_name: str
        Name of project eg. 'Jamaica'

    crs: str
        CRS to reproject all input data to and for model result

    landslide_points: str
        Path to landslide points file (takes geoJSON or shapefile)

    bbox: list
        Bounding box. Required for downloading or clipping online datasets.
        Takes in a list of coordinates in WGS84, or a shapefile, or a geoJSON.
        Example: [[x1,y1], [x1,y2], [x2,y2], [x2,y1], [x1,y1]]

    random_state: int
        Takes an int of any value to determine a (reproducible) state of randomness.
        Determining the random state allows for results to be replicated if needed.

    inputs: dict
        Dictionary of input data pointing to paths or decision to include
        it in the model. An example of how to define inputs is provided
        in ``example.py``.

    no_data: list
        list of potential no_data values. Preferably, you should define
        a constant no_data value for all your input datasets, so that valid
        data will NOT be accidentally masked out. No warranty is provided
        for erroneous results if any no data values are a valid value in
        another dataset.

    pixel_size: int
        Resolution of pixel size. WARNING - pixel size is only relevant
        for datasets obtained online. The pixel size will therefore
        dictate the resolution that the (relevant) input data will be in.

    kernel_size: int = 3
        To account for potential uncertainty in landslide-striken areas,
        a default [3 x 3] window of pixels around landslide points
        is classed as 'landslide'. The user can define a larger kernel size
        of ODD numbers eg. 3, 5, 7, 9 etc, depending on the resolution
        of their input datasets.

    Returns
    -------
    :object:`pandas.DataFrame`:
        Object is returned containing columns of input data as defined by
        the user [x] as well as another pandas.DataFrame object of classes.
        Each row represents a pixel index in the stack of input datasets.

    '''

    def __init__(self,
                 project_name: str,
                 crs: str,
                 landslide_points: str,
                 random_state: int,
                 bbox: list,
                 inputs: dict,
                 no_data: list = None,
                 pixel_size: int = None,
                 kernel_size: int = 3):

        # Setting up project directory and creating input, intermediate and output
        # folders
        self.project_name = project_name
        self.project_dir = pathlib.Path(f'./Projects/{self.project_name}')

        self.input_dir = self.project_dir / 'Input'
        self.int_dir = self.project_dir / 'Intermediate'
        self.output_dir = self.project_dir / 'Output'

        for i in [self.project_dir, self.input_dir, self.int_dir, self.output_dir]:
            if not i.exists():
                i.mkdir(parents = True)

        # Dictate crs to be used for output and grid alignment
        self.crs = crs


        # Checks if landslide points file exists and reads it
        assert pathlib.Path(landslide_points).exists(), 'Check your landslide points file path'

        self.landslide_points = gpd.read_file(landslide_points)

        # Checks if landslide points has valid crs for reprojection
        assert self.landslide_points.crs, 'No CRS assigned to landslide points, check projection'
        if self.landslide_points.crs != self.crs: # Reprojects landslide points to intended crs
            self.landslide_points = self.landslide_points.to_crs(crs = self.crs)


        # Bounding box checks
        assert bbox, 'Missing boundaries, please define a bounding box or input a shapefile/GJSON path'

        if type(bbox) == list:
            self.bbox = bbox

        elif bbox.endswith('.shp') or bbox.endswith('.json'):
            gdf = gpd.read_file(bbox)
            self.bbox = gdf.total_bounds

        self.kernel_size = kernel_size
        self.pixel_size = pixel_size

        def ee_load(param):
            '''
            load data from Earth Engine if no input provided
            bbox is only a list for now but may be in .shp or .json. Make it flexible.
            '''
            print(f'Downloading {param}...')
            inputs = {
                'dem': ee.Image("USGS/SRTMGL1_003").select('elevation'),
                #'slope': ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003").select('elevation')),
                #'aspect': ee.Terrain.aspect(ee.Image("USGS/SRTMGL1_003").select('elevation')),
                #'tpi': ee.Image("CSP/ERGo/1_0/Global/SRTM_mTPI").select('elevation'),
                #'landcover': ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019").select('discrete_classification')
                }

            bb_filter = inputs[param].clip(ee.Geometry.Polygon(bbox))

            return getURL(bb_filter, param)

        def getGLIM():
            wcs = WebCoverageService(r'https://deltaresdata.openearth.eu/geoserver/chw2/wcs?service=WCS',
                                     version = '1.0.0')

            # Get min/max of lon/lat
            minX, minY = np.array( self.bbox ).min(axis = 0)
            maxX, maxY = np.array( self.bbox ).max(axis = 0)

            # convert to EPSG:3857 and round to nearest thousandth
            trx = Transformer.from_crs(crs_from = 'EPSG:4326',
                                       crs_to = 'EPSG:3857',
                                       always_xy = True)

            # Reproject to EPSG:3857, and get floor/ceiling of nearest 1000th
            # place so that grid is regular. 1000th place because GLIM has 1km res.
            minX, minY = trx.transform(minX, minY)
            minX -= (minX % 1000); minY -= (minY % 1000)

            maxX, maxY = trx.transform(maxX, maxY)
            maxX -= (maxX % -1000); maxY -= (maxY % -1000)

            # bbox specific to GLIM's crs
            glimbbox = (minX, minY, maxX, maxY)

            r = wcs.getCoverage(identifier = 'chw2:glim_raster',
                                bbox = glimbbox, format='GeoTIFF',
                                crs = 'urn:ogc:def:crs:EPSG::3857',
                                resx=1000, resy=1000)

            with open(self.input_dir / 'glim.tif', 'wb') as f:
                f.write(r.read())

            # now you have to clean GLIM up, GLIM has no data values too.
            #dx = rx.open_rasterio(self.input_dir / 'glim.tif', lock = False); dx.close()

            return self.input_dir / 'glim.tif'

        def getSlopeAspect(param):
            print(f'Calculating {param} from DEM...')

            dem = rasterio.open(self.dem.path, lock=False, mask=True)

            elevation = dem.read(1)

            file = rasterio.open(self.input_dir / f'{param}.tif',
                                'w',
                                driver = 'GTiff',
                                height = dem.height,
                                width = dem.width,
                                count = 1,
                                dtype = np.float64,
                                crs = dem.crs,
                                transform = dem.transform)

            dx, dy = np.gradient(elevation)

            slope = (np.pi / 2) - np.arctan(np.sqrt(dx**2 + dy**2))
            aspect = np.arctan2(-dx, dy)
            rad2deg = 180.0 / np.pi

            if param == 'slope':
                file.write(slope*rad2deg, 1)
            elif param == 'aspect':
                file.write(aspect*rad2deg, 1)
            file.close; dem.close()

            return self.input_dir / f'{param}.tif'

        def getURL(img,
                   param):
            '''
            Gets download URL of images and downloads automatically to local folder.
            This option is preferred if your extent is not too large. Otherwise, please
            export to you G-Drive or assets and then manually download it.
            It won't run a check on whether you have the files already as the files
            can have a different scale / crs.
            '''

            link = img.getDownloadURL({
                'scale':        self.pixel_size,
                'crs':          self.crs,
                'region':       ee.Geometry.Polygon(self.bbox)
            })

            filepath = self.input_dir / f'{param}.tif'
            print(filepath)

            if filepath.exists():
                filepath.unlink()

            open_link = urllib.request.urlopen(link)
            zipfile = open_link.info().get_filename()

            r = urllib.request.urlretrieve(link,
                                           self.input_dir /
                                           open_link.info().get_filename())

            with ZipFile(self.input_dir / zipfile, 'r') as zipObj:
                if len(zipObj.namelist()) > 1:
                    pass #stop the script, warning exception raised.
                else:
                    zipObj.extractall(self.input_dir)
                    (self.input_dir / zipObj.namelist()[0]).rename(filepath)
            (self.input_dir / zipfile).unlink()
            if param == 'dem':
                file = rasterio.open(filepath, mode = 'r+')
                dem = file.read(1)
                dem = np.where(dem == -32767, -9999, dem)

                file.nodata = -9999
                file.write(dem, 1)
                file.close()
            return filepath

        def alignGrid(self, param):

            open_ref = rx.open_rasterio(self.reference.path); open_ref.close()
            crs_no = self.crs.split('epsg:')[1]

            # First ensure that reference dataset is in correct crs, if not, reproject.
            if not open_ref.rio.crs == self.crs:
                open_ref = open_ref.rio.reproject(self.crs)
                refname = self.reference.path.name.split('.tif')[0]
                open_ref.rio.to_raster(self.input_dir / f'{refname}_{crs_no}.tif')
                self.reference.path = self.input_dir / f'{refname}_{crs_no}.tif'


            open_param = rx.open_rasterio(param.path, lock=False); open_param.close()

            open_param = open_param.rio.reproject_match(open_ref,
                                                        resampling = Resampling.nearest)
            # or bicubic?
            open_param.rio.to_raster(self.input_dir / f'{param.name}_{crs_no}.tif')

            param.path = self.input_dir / f'{param.name}_{crs_no}.tif'

            return

        def alignShp(self, param):

            # checks if shapefile is in the right CRS

            open_file = gpd.read_file(param.path)
            if not open_file.crs == self.crs:
                open_file = open_file.to_crs(self.crs)
                open_file.to_file(param.path)
                del open_file
            else:
                del open_file

            return

        # valid_data = ['dem', 'slope', 'aspect', 'lithology', 'landcover']

        online_sources = {'dem': '''ee_load('dem')''',
                          'lithology': '''getGLIM()''',
                          'landcover': '''ee_load('landcover')''',
                          'slope': '''getSlopeAspect('slope')''',
                          'aspect': '''getSlopeAspect('aspect')'''}
        model_inputs = {}

        for key, value in inputs.items():

            if key == 'reference':
                self.reference = getattr(self, value)

                # check and assign reference attributes
                if not all([hasattr(self.reference, i) for i in ['height', 'width',
                                                                 'xsize', 'ysize']]):

                    open_ref = rasterio.open(self.reference.path, lock=False)
                    self.reference.width = open_ref.width
                    self.reference.height = open_ref.height

                    self.reference.xsize = open_ref.get_transform()[1]
                    self.reference.ysize = -(open_ref.get_transform()[-1])

            elif value['filepath'] == None:
                continue

            # elif not key in valid_data:
            #     warnings.warn(
            #         f'Dataset {key} not in the following list '
            #         f'{valid_data} .'
            #         'Dataset will be excluded from model input.')

            elif pathlib.Path(value['filepath']).exists():
                setattr(self, key, inputData(key, value['filepath'], value['data_type']))
                model_inputs.update({key: getattr(self, key)})

            elif value['filepath'] == 'online':
                setattr(self, key, inputData(key, eval(online_sources[key]), value['data_type']))
                model_inputs.update({key: getattr(self, key)})

        self.model_inputs = model_inputs

        for i, a in model_inputs.items():
            if self.reference == a:
                continue
            alignGrid(self, a) # Aligns all the grid to reference extent.

        self.no_data = no_data

        inputs_message = (
            'All inputs loaded, aligned and ready to run. To check '
            'model inputs, call project.model_inputs')

        print(inputs_message)

    def valid_arrays(self):
         '''
         Generates list of valid arrays. A mask is made of only valid arrays
         across stack of arrays.

         '''
         self.arrays = []
         self.names = []

         for k, v in self.model_inputs.items():

             dd = rasterio.open(v.path, lock=False, mask = True)

             # GeoTIFFs have a maximum of 4 bands, __.read(1) reads the 1st band
             # number

             # Append all arrays to list to create stack. Float32 not yet tested.
             self.arrays.append(dd.read(1).astype(np.float64))
             dd.close()
             # Append all names of arrays into list of names. This can also  be
             # accessed with project.model_inputs when running the tool.
             self.names.append(k)

         # Filters no data values
         self.arrays = np.where(np.isin(self.arrays, self.no_data), np.nan, self.arrays)

         # Creates a mask over all nan values and subsequently tiles it to similar
         # shape as raster stack.
         mask = np.tile((~np.any(np.isnan(self.arrays),
                                 axis=0) & np.any(self.arrays, axis=0)),
                         (len(self.arrays),1,1))

         self.arrays = np.where(mask, self.arrays, np.nan)

         return

    def matrix_window(self,
               x, y, ID):

         '''
         kernel size is a default value of 3: can only be an odd number
         Generates a dataframe of x and y coordinates
         '''

         dd = rasterio.open(self.reference.path, lock = False)

         kernel_extent = self.kernel_size // 2

         xi = np.arange(x - kernel_extent, x + kernel_extent + 1, 1)
         yi = np.arange(y - kernel_extent, y + kernel_extent + 1, 1)

         _xi, _yi = np.meshgrid(xi, yi)
         ii_list = [[a, b, ID] for a, b in zip(_xi.flatten(), _yi.flatten())]

         dd.close()

         return ii_list

    def generate_xy(self):
        '''
        Takes the landslide points and selects pixels that overlap with the landslide points
        as well as the matrix area around it. Kernel is definable by user.

        Returns a dataframe of raster values in those indexes

        to-do: use random_State attribute for sampling
        '''
        self.valid_arrays()

        # Coordinates
        dd = rasterio.open(self.reference.path, lock = False)

        points = [(a, b) for a, b in zip(self.landslide_points.geometry.x.tolist(),
                                          self.landslide_points.geometry.y.tolist())]

        #df = pd.DataFrame(columns = ['x', 'y', 'id'])
        iid = [] #Generate list of index tuples

        for idx, pts in enumerate(points):
            row, col = dd.index(pts[0], pts[1])
            idd = self.matrix_window(row, col, idx)
            iid += idd

        dd.close()

        # Create an empty boolean mask
        bmask = np.full_like(self.arrays[0], False, dtype=bool)

        # Create an array of landslide IDs
        idarray = np.full_like(self.arrays[0], np.nan, np.float64)

        # Assign True to points in the raster stack that are considered landslide points
        for i in iid:
            bmask[i[0], i[1]] = True
            idarray[i[0], i[1]] = i[2]

        # Add the id array to stack of arrays
        self.arrays = np.append(self.arrays, [idarray], axis=0)
        self.names += ['landslide_ids']
        # Create same shape of boolean mask as the raster stack
        bmasks = np.tile(bmask, (len(self.arrays), 1, 1))

        '''
        self.arrays[bmasks] == filtering landslide pixels in the raster stack
        Rest of the steps to prepare data for ingestion into dataframe
        '''
        lx = self.arrays[bmasks].reshape( len(self.arrays), 1, -1).transpose().tolist()
        lx = [i[0] for i in lx] # Prepare for ingestion to dataframe

        self.landslide_pixels = pd.DataFrame.from_records(lx,
                                                          columns = self.names)
        self.landslide_pixels['id'] = 1 # Assign 1 as landslide points
        self.landslide_pixels = self.landslide_pixels.dropna()

        nonlx = self.arrays[~bmasks].reshape(len(self.arrays), 1, -1).transpose().tolist()
        nonlx = [i[0] for i in nonlx]

        nldf = pd.DataFrame.from_records(nonlx,
                                         columns = self.names)
        nldf = nldf.drop(columns=['landslide_ids']).dropna()

        self.nonlandslide_pixels = nldf.sample(n = len(self.landslide_pixels.index))
        self.nonlandslide_pixels['id'] = 0

        self.model_input = pd.concat([self.landslide_pixels,
                                      self.nonlandslide_pixels],
                                     sort=False).reset_index(drop=True)

        #self.model_input = self.model_input.dropna().reset_index()
        self.x = self.model_input[self.model_input.columns[~self.model_input.columns.isin(['id'])]]
        self.y = self.model_input.id

        print(
            'CAUTION\n'
            'Landslide IDs are included in the input dataframe (x).\n'
            'Drop the landslide IDs before running the model.'
            )
        return self.x, self.y

    def run_model(self,
                  model: str,
                  x,
                  y,
                  modelExist: bool):
        '''
        model choices:  ['SVM', 'RF', 'LR']. Default: 'SVM' (Support Vector Machine)
        '''
        self.model = model

        lx_model = md.MachineLearning(x, y, self.int_dir, model_name = model,
                                   modelExist = False)

        print('Training and predicting values...')

        # Arrays are taken but the last ID array is skipped of course
        lx_model.predict_proba(self.arrays[:-1,:,:], estimator = lx_model.bestModel,
                               scaler = lx_model.scaler,
                               file_path = self.output_dir / f'result_{model}.tif',
                               reference = self.reference.path,
                               no_data = self.no_data)

        print('Model complete. Check output folder for results.')

        return

    def vector2raster(self,
                      vectorpath):

        # open shapefile, get layer and get extent
        ds = ogr.Open(vectorpath)
        lyr = ds.GetLayer()

        xmin, xmax, ymin, ymax = lyr.GetExtent()

        cols = math.ceil((xmax - xmin) / self.reference.width)
        rows = math.ceil((ymax - ymin) / self.reference.height)

        # create empty output raster
        out_fn = self.input_dir / f'{pathlib.Path(vectorpath).stem}.tif'
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(str(out_fn), cols, rows, 1, gdal.GDT_Byte)

        # set geotransform and projection
        gt = (xmin, self.reference.xsize, 0, ymax, 0, -(self.reference.ysize))
        sr = lyr.GetSpatialRef().ExportToWkt()
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(sr)

        # rasterize layer
        band = out_ds.GetRasterBand(1)
        gdal.RasterizeLayer(out_ds, [1], lyr, burn_values=[1])

        # set NoData value and close shapefile and raster
        band.SetNoDataValue(self.no_data)
        del ds, out_ds
        return out_fn

    def proximity2feature(self,
                      rastervec: str):
        '''
        '''
        # open rasterized file and get information
        ds = gdal.Open(rastervec, 0)
        band = ds.GetRasterBand(1)
        gt = ds.GetGeoTransform()
        sr = ds.GetProjection()
        cols = ds.RasterXSize
        rows = ds.RasterYSize

        # create empty proximity raster
        out_fn = self.input_dir / f'{pathlib.Path(rastervec.stem)}_proximity.tif'
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_fn, cols, rows, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(sr)
        out_band = out_ds.GetRasterBand(1)

        # compute proximity
        # Caution: Ensure your data are in UTM coordinates or distance
        # will be wrongly calculated!
        gdal.ComputeProximity(band, out_band, ['VALUES=1', 'DISTUNITS=GEO'])

        # delete input and output rasters
        del ds, out_ds
        return out_fn
