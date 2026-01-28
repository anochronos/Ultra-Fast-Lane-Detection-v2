# BDD100K Configuration for UFLD v2
# Based on TuSimple config but adapted for BDD100K dataset

dataset = 'BDD100K'
data_root = 'data/bdd100k'  # Relative to repo root

# Training settings
epoch = 100
batch_size = 32
optimizer = 'SGD'
learning_rate = 0.05
weight_decay = 0.0001
momentum = 0.9
scheduler = 'multi'
steps = [50, 75]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100

# Model settings
backbone = '18'
griding_num = 100
use_aux = False

# Loss settings
sim_loss_w = 0.0
shp_loss_w = 0.0
mean_loss_w = 0.05
var_loss_power = 2.0

# Logging
note = ''
log_path = ''

# Checkpoints
finetune = None
resume = None
test_model = ''
test_work_dir = ''
auto_backup = True

# Lane settings
num_lanes = 4

# Row/Col settings (similar to TuSimple since BDD100K has similar resolution)
num_row = 56
num_col = 41

# Image size settings
# BDD100K native resolution is 1280x720
# We resize to 800x288 for training (similar to TuSimple)
train_width = 800
train_height = 320

num_cell_row = 100
num_cell_col = 100

# Other settings
fc_norm = False
soft_loss = True
cls_loss_col_w = 1.0
cls_ext_col_w = 1.0
mean_loss_col_w = 0.05
eval_mode = 'normal'
crop_ratio = 0.6
