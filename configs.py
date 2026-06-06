BASE_PATH = 'Data/VCTKCorpus'
seed = 67
actfe = actfd = actfs = 'relu'

low_sampling_rate = 8000
high_sampling_rate = 24000
val_scale = 2

limit = None
clip_size_sec = 0.5
batch_size = 8
update_step = 4

lr = 1e-3
epochs = 60
step_size = 12
gamma = 0.5
wdc = 1e-4

max_norm = 0.1
mssl_wt = 1
l1_wt = 55
var_wt = 300
g_wt = 10

omega = 50
num_bands = 8
mdim = 256