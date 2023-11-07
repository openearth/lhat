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
    project_name = 'jamaica_test',

    # The crs defined here will dictate which crs your input data is reprojected
    # to, as well as your final result.
    crs = 'epsg:3450',

    # Provide a path to your landslide points. This is COMPULSORY for the model
    # to work.
    landslide_points = './Projects/jamaica-test/Input/dummy-landslides.json',

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
    inputs = {
        'dem': {'filepath': 'online',
                'data_type': 'numerical'},
        'slope': {'filepath': 'online',
                  'data_type': 'numerical'},
        'aspect': {'filepath': 'online',
                    'data_type': 'numerical'},
        'lithology': {'filepath': 'online',
                        'data_type': 'categorical'},
        'prox_road': {'filepath': ".\Projects\jamaica-test\Input\prox_roads.tif",
                      'data_type': 'numerical'},
        'prox_river': {'filepath': ".\Projects\jamaica-test\Input\prox_rivers.tif",
                        'data_type': 'numerical'},
        'reference': 'dem'
        },

    no_data = -9999,  # Optional argument to define no_data value. Propogates
                        # for all processing of input files.

    pixel_size = 1000,    # Optional argument to define pixel size.
                          # Pixel size is only important for online datasets

    kernel_size = 3     # Define kernel size. Take into consideration pixel size
                        # and full extent of landslide-prone areas.
    )

######   Data Engineering Stage   ######
# The user has a choice to further refine the input data prior to running the
# model.
x, y = project.generate_xy()

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