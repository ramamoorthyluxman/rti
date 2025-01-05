import numpy as np
import os
import glob
import math
from . import params
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


# Convert Cartesian coordinates to spherical coordinates    
def Cartesian2spherical3D(x, y, z):
    """
    Takes X, Y, and Z coordinates as input and converts them to a spherical
    coordinate system

    Source: https://stackoverflow.com/questions/10868135/cartesian-to-spherical-3d-coordinates

    """

    r = math.sqrt(x*x + y*y + z*z)

    longitude = math.acos(x / math.sqrt(x*x + y*y)) * (-1 if y < 0 else 1)

    latitude = math.acos(z / r)

    return r, longitude, latitude

# Convert spherical coordinates to Cartesian coordinates
def spherical2Cartesian3D(r, longitude, latitude):
    """
    Takes, r, longitude, and latitude coordinates in a spherical coordinate
    system and converts them to a 3D cartesian coordinate system

    Source: https://stackoverflow.com/questions/10868135/cartesian-to-spherical-3d-coordinates
    """

    x = r * math.sin(latitude) * math.cos(longitude)
    y = r * math.sin(latitude) * math.sin(longitude)
    z = r * math.cos(latitude)

    return x, y, z

class dataset:
    def __init__(self, acq_path):
        self.acqs = acq_path
        self.lps_cartesian = []
        self.lps_spherical = []
        self.image_paths = []
        self.save_paths = []
        self.azimuths = []
        self.elevations = []
        self.pixel_wise_lps = []
        self.images = []

        
        # Find .lp file
        lp_files = glob.glob(os.path.join(acq_path, "*.lp"))
        if not lp_files:
            raise ValueError(f"No .lp file found in {acq_path}")
        self.lp_file = lp_files[0]  # Take the first .lp file

        self.load_lps_and_img_paths()

        if params.TRAINING:
            print("Loading images...")
            self.load_images()

    def load_lps_and_img_paths(self):
        print("Loading LP file and image paths...")
    
        with open(self.lp_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.split()
                if len(parts) != 4:
                    print(f"Line in {self.lp_file} does not contain exactly four parts: {line}")
                    return

                img_file, x, y, z = parts
                x = float(x)
                y = float(y)
                z = float(z)
                self.lps_cartesian.append((x, y, z))
                self.lps_spherical.append(Cartesian2spherical3D(x, y, z))
                _, azimuth, elevation = Cartesian2spherical3D(x, y, z)
                self.azimuths.append(azimuth)
                self.elevations.append(elevation)
                self.image_paths.append(os.path.join(os.path.dirname(self.lp_file), img_file))
                self.save_paths.append(os.path.join(params.SAVE_PATH, img_file))
            self.lps_cartesian = np.array(self.lps_cartesian)
            self.lps_spherical = np.array(self.lps_spherical)
            self.azimuths = np.array(self.azimuths)
            self.elevations = np.array(self.elevations)
            self.image_paths = np.array(self.image_paths)
    
    def load_images(self):
        for img_path in tqdm(self.image_paths, desc="Loading images", unit="img"):
            img = cv2.imread(img_path)
            self.images.append(img)
        self.images = np.array(self.images)       


    def compute_pixel_wise_lps(self):
        # to be implemented
        return

    def create_and_save_heatmaps(self):
        # to be implemented
        return 