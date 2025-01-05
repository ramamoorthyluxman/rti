from ptm import PTMModel
from hsh import HSHModel
from utils import dataset
from utils import params
import numpy as np

data_ = dataset(params.ACQ_PATH)

target_images = np.array(data_.images)
azimuths = np.array(data_.azimuths)
elevations = np.array(data_.elevations)

save_paths = data_.save_paths

print("Target images shape: ", target_images.shape)
print("Azimuths shape: ", azimuths.shape)
print("Elevations shape: ", elevations.shape)

# For PTM
# ptm = PTMModel()
# coeffs = ptm.model_fit(azimuths, elevations, target_images)
# relit_images = ptm.relight(azimuths=azimuths, 
#                            elevations=elevations, 
#                            target_images=target_images, 
#                            save_paths=save_paths)

# For HSH
hsh = HSHModel()
coeffs = hsh.model_fit(azimuths, elevations, target_images)
relit_images = hsh.relight(azimuths=azimuths, 
                           elevations=elevations, 
                           target_images=target_images, 
                           save_paths=save_paths)