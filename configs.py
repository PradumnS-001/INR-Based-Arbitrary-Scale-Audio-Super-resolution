BASE_PATH = 'Data/VCTKCorpus'
seed = 42
actfe = actfdc = actfdi = actfs = 'silu'
low_sampling_rate = 8000
high_sampling_rate = 24000
val_scale = 2

limit = 20480
clip_size_sec = 0.5
batch_size = 8
update_step = 4
scale_res = 100

base_weights={
        'mssl': 2.0, 'huber': 1.0,
        'hinge_1x': 1.0, 'fm_1x': 2.0,
        'hinge_2x': 1.0, 'fm_2x': 2.0,
        'hinge_3x': 1.0, 'fm_3x': 2.0
    }

lr = 1e-4
epochs = 50
step_size = 10
gamma = 0.1
wdc = 1e-4
scheduler_start = 0
thershold = 0.6

max_norm = 1
encoder_dim = 32
filters = 22
freq_bands = 16