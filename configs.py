import numpy as np

BASE_PATH = 'Data/VCTKCorpus'
seed = 42
actfe = actfd = actfs = 'elu'

low_sampling_rate = 8000
high_sampling_rate = 24000
val_scale = 2
temperature = 0.7

limit = None
clip_size_sec = 0.5
batch_size = 4
update_step = 8
scale_res = 50

lr = 1e-3
epochs = 50
step_size = 6
gamma = 0.2
wdc = 1e-5
scheduler_start = 0

max_norm = 1
l1_wt = 0.05
dist_wt = 2
mssl_wt = 3
log_loss_eps = 1e-3

is_omega_trainable = False
omega = np.pi
num_bands = 6
mdim = 64

num_blocks = 5
noisy_start = 2
drop_prob = 0.15
do_perturbation = True