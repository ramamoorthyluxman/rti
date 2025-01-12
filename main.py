from ptm import PTMModel
from hsh import HSHModel
from dmd import DMDModel
from utils import dataset
from utils import params
import numpy as np

from overlap_analysis_utils import apply_overlap_analysis

data_ = dataset(params.ACQ_PATH)

target_images = np.array(data_.images)

lps_cartesian = np.array(data_.lps_cartesian)

save_paths = data_.save_paths

print("Target images shape: ", target_images.shape)

print("Shape of LPs cartesians: ", lps_cartesian.shape)

# For PTM
# ptm = PTMModel()
# coeffs = ptm.model_fit(lps_cartesian=lps_cartesian, target_images=target_images)
# relit_images = ptm.relight(lps_cartesian, 
#                            target_images=target_images, 
#                            save_paths=save_paths)

# For HSH
# hsh = HSHModel()
# coeffs = hsh.model_fit(lps_cartesian=lps_cartesian, target_images=target_images)
# relit_images = hsh.relight(lps_cartesian=lps_cartesian, 
#                            target_images=target_images, 
#                            save_paths=save_paths)

# For DMD
dmd = DMDModel()
coeffs = dmd.model_fit(lps_cartesian=lps_cartesian, target_images=target_images)
relit_images = dmd.relight(lps_cartesian=lps_cartesian,
                           target_images=target_images, 
                           save_paths=save_paths)