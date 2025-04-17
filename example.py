################################################
#####           PARAMETER SET-UP           #####
################################################
# Set-up of Landslide Hazard Assessment Tool (LHAT)
#
# The tool initialises and takes the following arguments
# necessary to parameterise both data preparation, alignment
# and data extraction steps.
# The following is a working example

from lhat import IO as io

project = io.inputs(

    # Define a project name. This will be the name of the folder in which
    # your results are stored in
    project_name = 'Liguria_test',

    # The crs defined here will dictate which crs your input data is reprojected
    # to, as well as your final result.
    crs = 'epsg:3044',

    # Provide a path to your landslide points. This is COMPULSORY for the model
    # to work.
    landslide_points = './Projects/Liguria_test/Input/flash_floods_Liguria.json',

    # Defining a random state (any integer) allows results to be reproducible
    random_state = 101,

    # A bounding box is required when taking inputs from online sources such as
    # geoservers. Use EPSG:4326 coordinates.
    bbox = [[-77.73174142, 18.02046626],
            [-77.1858101, 18.02046626],
            [-77.1858101, 18.34868174],
            [-77.73174142, 18.34868174],
            [-77.73174142, 18.02046626]],

    # The following are inputs that are possible to use within LHAT.
    # 3 choices for filepaths are: your_file_path, 'online', None.
    #       your_file_path = path to the respective file in string
    #       'online'       = an online, typically global source is relied on instead.
    #                        For datasets that are calculated from another dataset
    #                        such as slope/aspect/roughness, leave as 'online'.
    #       None           = None as an argument means that the dataset is NOT
    #                        considered as input into the model.
    #
    # Data type is critical to define as categorical and numerical data undergo
    # different data treatments.
    #
    # For 'reference', take care that if an online dataset is used as the reference,
    # bbox arguments define the grid extent, while the pixel_size argument below
    # defines the resolution of your reference (and therefore, your output) dataset.
    # inputs = {
    #     'dem': {'filepath': 'online',
    #             'data_type': 'numerical'},
    #     'slope': {'filepath': 'online',
    #               'data_type': 'numerical'},
    #     'aspect': {'filepath': 'online',
    #                 'data_type': 'numerical'},
    #     # 'lithology': {'filepath': 'online',
    #     #                 'data_type': 'categorical'},
    #     'prox_road': {'filepath': ".\Projects\jamaica-test\Input\prox_roads.tif",
    #                   'data_type': 'numerical'},
    #     'prox_river': {'filepath': ".\Projects\jamaica-test\Input\prox_rivers.tif",
    #                     'data_type': 'numerical'},
    #     'reference': 'dem'
    #     },

    inputs = {
        'dem': {'filepath': '.\Projects\Liguria_test\Input\dem_3044_Liguria.tif',
                'data_type': 'numerical'},
        # 'hand': {'filepath': '.\Projects\Liguria_test\Input\hand_3044_Liguria.tif',
        #          'data_type': 'numerical'},
        'landcover': {'filepath': '.\Projects\Liguria_test\Input\landcover_3044_Liguria.tif',
                'data_type': 'categorical'},
        # 'lithology': {'filepath': '.\Projects\Liguria_test\Input\lithology_3044_Liguria.tif',
        #         'data_type': 'categorical'},
        # 'aspect': {'filepath': r'.\Projects\Liguria_test\Input\aspect_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        'curvature': {'filepath': '.\Projects\Liguria_test\Input\curvature_3044_Liguria.tif',
                'data_type': 'numerical'},
        # 'slope': {'filepath': '.\Projects\Liguria_test\Input\slope_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        # 'ap': {'filepath': r'.\Projects\Liguria_test\Input\prec_ap_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        # 'NDVI2017': {'filepath': r'.\Projects\Liguria_test\Input\ndvi_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        # '3hp': {'filepath': r'.\Projects\Liguria_test\Input\prec_3hp_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        '6hp': {'filepath': r'.\Projects\Liguria_test\Input\prec_6hp_3044_Liguria.tif',
                'data_type': 'numerical'},
        # '12hp': {'filepath': r'.\Projects\Liguria_test\Input\prec_12hp_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        # '24hp': {'filepath': r'.\Projects\Liguria_test\Input\prec_24hp_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        # 'twi': {'filepath': r'.\Projects\Liguria_test\Input\twi_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        # 'spi': {'filepath': '.\Projects\Liguria_test\Input\spi_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        # 'prox_river': {'filepath': '.\Projects\Liguria_test\Input\prox_rivers_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        # 'prox_road': {'filepath': '.\Projects\Liguria_test\Input\prox_roads_3044_Liguria.tif',
        #         'data_type': 'numerical'},
        'reference': 'dem'
    },

    no_data = -9999,  # Optional argument to define no_data value. Propogates
                        # for all processing of input files.

    pixel_size = 1000,    # Optional argument to define pixel size.
                          # Pixel size is only important for online datasets

    kernel_size = 3,     # Define kernel size. Take into consideration pixel size
                        # and full extent of landslide-prone areas.
    )

######   Data Engineering Stage   ######
# The user has a choice to further refine the input data prior to running the
# model.
x, y = project.generate_xy()

# TODO: include a no data value, now the smallest value considered for DEM is 1 and not 0!
project.natural_breaks(x)

# get frequency ratio and thresholds for categorical data
df_FR_categorical = project.frequency_ratio(
    x = x,
    y = y,
    data_type='categorical'
)

# get frequency ratio and thresholds for numerical data
df_FR_numerical, df_thresholds = project.frequency_ratio(
    x = x,
    y = y,
    data_type='numerical'
)

# x = x.drop(columns=['landslide_ids'])

# As an example
# for m in ['SVM', 'RF', 'LR']:
#     project.run_model(
#         x = x,
#         y = y,
#         model = m,
#         modelExist = False
#         )

project.run_model(
    x = x,
    y = y,
    model = 'LR',
    modelExist = False
    )