# ACQ_PATHS = [r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_1/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_2/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_3/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_4/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_1_5/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_1/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_2/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_3/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_4/images/Face_A/rti',
#              r'/work/imvia/ra7916lu/illumi-net/data/retrato_de_lola_flores/dome/2024_02_22_2_5/images/Face_A/rti'
#             ]

ACQ_PATH = r'/work/imvia/ra7916lu/rti/data/buddhaPNG'
SAVE_PATH = r'/work/imvia/ra7916lu/rti/data/buddhaPNG/relight/dmd'


             
            
# Dataset params
MAX_NB_IMAGES_PER_ACQ = 105
COMPUTE_NORMALS_AND_ALBEDO = False # Ensure TRAINING is True
COMPUTE_DISTANCES_AND_COSINES = False # Ensure TRAINING is True
CREATE_DIST_COSINES_HEATMAPS = False # Ensure TRAINING is True
COLLIMATED_LIGHT = True 

# SURFACE_PHYSCIAL_SIZE = [(1024*0.250)/2704, (1024*0.160)/1800] #default [0.250, 0.120]
# SURFACE_PHYSCIAL_SIZE = [0.250, 0.160] #default [0.250, 0.160] #width , height
SURFACE_PHYSCIAL_SIZE = [0.05, 0.0421] #default [0.250, 0.120]


# Compute normals params
# PS_METHOD = "L2_SOLVER"    # Least-squares
# PS_METHOD = "L1_SOLVER_MULTICORE"    # L1 residual minimization
# PS_METHOD = "SBL_SOLVER_MULTICORE"    # Sparse Bayesian Learning
PS_METHOD = "RPCA_SOLVER"    # Robust PCA


# RTI training
RTI_NET_EPOCHS = 1000
RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS = 1
# RTI_NET_PATCH_SIZE = [256,256] #[height, width]. Must be divisible by 32  or [192, 192], [256, 256], etc.
RTI_NET_PATCH_SIZE = [64,64] #[height, width]. Must be divisible by 32  or [192, 192], [256, 256], etc.
RTI_MAX_NUMBER_PATCHES = 128 # Randomly choose desired number of patches from the image. 

# RTI Relighting
RTI_MODEL_PATH = r'/work/imvia/ra7916lu/illumi-net/saved_models/saved_models_43_20241113_225334/relighting_model_epoch_500.pth'
RTI_MODEL_SAVE_DIR = r'/work/imvia/ra7916lu/illumi-net/data/subset/2024_02_22_1_3/images/Face_A/rti_sub_images/results/illumi-net/model'

# Goal
TRAINING = True # Training: True, Relighting: False. If you choose relighting- ensure the path contains - 1. distances.npy, 2. cosines.npy, 3. albedo and 4. normals


# Model Params

BATCH_SIZE = 4
TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
NUM_WORKERS = 4
LEARNING_RATE = 0.001
LAMBDA_MSE = 1.0
LAMBDA_L1 = 1.0
LAMBDA_HIGHLIGHT = 0.5
LAMBDA_GRADIENT = 0.5
LAMBDA_SPECULAR = 0.9
LAMBDA_CONTRAST = 0.5
LAMBDA_PERCEPTUAL = 1.2
LAMBDA_DARK = 0.5       # Weight for dark region accuracy
LAMBDA_MID = 0.1         # Weight for mid-tone accuracy
LAMBDA_BRIGHT =  0.8      # Weight for highlight accuracy
LAMBDA_LARGE_DEV = 0.1   # Weight for large deviation penalty

EPOCHS_PER_CYCLE = 10

PERSISTENT_WORKER =True,
PIN_MEMORY =True,
PREFETCH_FACTOR=3
CUDA_VISIBLE_DEVICES = "0,1"
PATCH_PIX_VAL_THRESHOLD = 0.1
NON_BLACK_PIX_RATIO_MIN = 0.5
RTI_MAX_IMAGES_PER_CHUNK = 40  # Maximum number of images to process at once. For relighting
TRAIN_VAL_SPLIT = 0.2

OPTIMIZER = "Adam"
WEIGHT_DECAY = 0.004136741236796869

# Visualizations
SAMPLE_NUMBER_OF_COMPARISONS = 8

# Hyperparameter optimization settings
OPTIMIZE_HYPERPARAMETERS = False  # Set to True when you want to run optimization
OPTIMIZATION_TRIALS = 100  # Number of trials for optimization