import numpy as np

BASE_PATH = 'Data/VCTKCorpus'
seed = 42
actfe = actfd = actfs = 'relu'

low_sampling_rate = 8000
high_sampling_rate = 24000
val_scale = 3

limit = None
clip_size_sec = 0.5
batch_size = 4
update_step = 8
scale_res = 50

lr = 1e-3
epochs = 50
step_size = 10
gamma = 0.1
wdc = 1e-5
scheduler_start = 0

max_norm = 0.1
l1_wt = 80
mssl_wt = 5
loss_eps = 1e-3
loss_pow_fac = 0.5
mdim = 128
mdim1 = 96
filters = 8

num_blocks = 4
noisy_start = 4
drop_prob = 0.125
do_perturbation = True