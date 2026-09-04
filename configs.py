BASE_PATH = 'Data/VCTKCorpus'
seed = 42
actfe = actfdc = actfdi = actfs = 'relu'
low_sampling_rate = 8000
high_sampling_rate = 24000
val_scale = 2
mdim = 196

limit = 10240
clip_size_sec = 0.6
batch_size = 8
update_step = 4
scale_res = 100

base_weights={
        'mssl': 3.0, 'huber': 3.0,
        'hinge_1x': 2.0, 'fm_1x': 2.0,
        'hinge_2x': 2.0, 'fm_2x': 2.0,
        'hinge_3x': 2.0, 'fm_3x': 2.0
    }

lr = 1e-3
epochs = 50
step_size = 10
gamma = 0.5
wdc = 1e-5
thershold = 0.5

max_norm = 1
encoder_dim = 32
filters = 20
freq_bands = 16

beta_steps = 1000
min_kld = 0.5
kld_wt = 0.05