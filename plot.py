# Author: Robyn S. Gwee
# Contact: robyn.gwee@deltares.nl
# Date created: Fri Oct  8 10:25:14 2021
# Remarks:

import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.plot import show
import numpy as np

tif_result = r"D:\EO\LandslideDetectionSO2021\master\Projects\Jamaica_test\Output\result_LR.tif"
landslide_pts = r"D:\EO\LandslideDetectionSO2021\master\Projects\Jamaica_test\Input\landslide.json"

pts = gpd.read_file(landslide_pts).to_crs('epsg:3450')


model = rasterio.open(tif_result, lock=False, mask=True)
maskmodel = np.where(model.read(2) == -9999, np.nan, model.read(2))

fig, ax = plt.subplots(1,1)
hidden = ax.imshow(maskmodel, cmap='Reds')
fig.colorbar(hidden, ax=ax, label='Probability of landslide occurrence',
             orientation='horizontal')
show(maskmodel, transform = model.transform, 
     cmap = 'Reds', ax=ax)
#pts.plot(marker='x', markersize=7, ax=ax, color= 'blue',alpha=0.1)