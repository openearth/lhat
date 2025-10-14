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

# folder = 'Liguria_flash_floods'
# folder='flash_floods'
folder='landslides'

project = io.inputs(

    # Define a project name. This will be the name of the folder in which
    # your results are stored in
    project_name = folder,

    # The crs defined here will dictate which crs your input data is reprojected
    # to, as well as your final result.
    crs = 'epsg:3044',

    # Provide a path to your landslide points. This is COMPULSORY for the model
    # to work.
    landslide_points= f'./Projects/{folder}/Input/landslides_v2.json',
    # landslide_points = f'./Projects/{folder}/Input/flash_floods_v3.geojson',

    # Defining a random state (any integer) allows results to be reproducible
    random_state = 101,

    # A bounding box is required when taking inputs from online sources such as
    # geoservers. Use EPSG:4326 coordinates.
    bbox = [[-77.73174142, 18.02046626],
            [-77.1858101, 18.02046626],
            [-77.1858101, 18.34868174],
            [-77.73174142, 18.34868174],
            [-77.73174142, 18.02046626]],


    inputs = {
        'dem': {'filepath': f'.\Projects\{folder}\Input\dem_3044_Liguria.tif',
                'data_type': 'numerical'},
        'prox_rivers': {'filepath': f'.\Projects\{folder}\Input\distance_to_rivers.tif',
                'data_type': 'numerical'},
        'landcover': {'filepath': f'.\Projects\{folder}\Input\landcover_Liguria.tif',
                'data_type': 'categorical'},
        'lithology': {'filepath': f'.\Projects\{folder}\Input\Lithology1_Liguria.tif',
                'data_type': 'categorical'},
        'aspect': {'filepath': rf'.\Projects\{folder}\Input\aspect_Liguria.tif',
                'data_type': 'numerical'},
        'curvature': {'filepath': f'.\Projects\{folder}\Input\curvature_Liguria.tif',
                'data_type': 'numerical'},
        'slope': {'filepath': f'.\Projects\{folder}\Input\slope_Liguria.tif',
                'data_type': 'numerical'},
        'NDVI2017': {'filepath': rf'.\Projects\{folder}\Input\NDVI2017_annual_4326_Liguria.tif',
                'data_type': 'numerical'},
        'twi': {'filepath': rf'.\Projects\{folder}\Input\twi_Liguria.tif',
                'data_type': 'numerical'},
        'spi': {'filepath': f'.\Projects\{folder}\Input\spi_Liguria.tif',
                'data_type': 'numerical'},
        'prox_roads': {'filepath': f'.\Projects\{folder}\Input\prox_road_3044.tif',
                'data_type': 'numerical'},
        
        'reference': 'dem'
    },

    no_data = -9999,  # Optional argument to define no_data value. Propagates
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


#%%

import math
binning_dict = {
    'dem': 4,
    'aspect': [-1, 0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi], # flat areas (no aspect),  NE, NW, SW, SE 
    'curvature': [-99999, -0.5, 0.5, 900],
    'prox_rivers': 4,
    'slope': 4,
    'NDVI2017': 4,
    'twi': 4,
    'spi': 4,
    'prox_roads': 4
}
# project.binning(binning_dict)

test = project.iterate_FR(binning_dict=binning_dict)

#%%
df_FR_categorical = project.frequency_ratio(
    x = x,
    y = y,
    data_type='categorical'
)

# export dataframes
df_FR_categorical.to_csv(f'.\Projects\{folder}\Output\FR_categorical.csv')
#%%

# x = x.drop(columns=['landslide_ids'])

# As an example
for m in ['LR', 'RF', 'SVM']:
    trained_model = project.run_model(
        x = x,
        y = y,
        model = m,
        modelExist = False
        )
    

#%%
model_names = [ 'LR', 'RF', 'SVM']
models = {}
scalers = {}

# plot ROC curves for multiple models
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

for m in model_names:
    models[m] = joblib.load(f'./Projects/{folder}/Intermediate/{m}_bestModel.sav')
    scalers[m] = joblib.load(f'./Projects/{folder}/Intermediate/{m}_scaler.sav')

X_test = joblib.load(f'./Projects/{folder}/Intermediate/X_test.sav')
y_test = joblib.load(f'./Projects/{folder}/Intermediate/y_test.sav')

plt.figure(figsize=(7, 5))
for m in model_names:
    y_pred_proba = models[m].predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_pred_proba, name=m, ax=plt.gca())
plt.title(f"ROC Curves for {folder}")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(f'./Projects/{folder}/Output/ROC_Curves.png', dpi=300)
plt.show()
# %%
