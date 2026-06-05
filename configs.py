BASE_PATH = 'Data/VCTKCorpus'
seed = 67
actfe = 'gelu'
actfd = 'gelu'
actfs = 'gelu'

low_sampling_rate = 8000
high_sampling_rate = 24000
val_scale = 2

limit = None
clip_size_sec = 0.5
batch_size = 32

lr = 1e-3
epochs = 50

gamma = 0.5
step_size = 10

max_norm = 0.1
mssl_wt = 1
l1_wt = 500

omega = 100
num_bands = 8