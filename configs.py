BASE_PATH = 'Data/VCTKCorpus'
seed = 42
low_sampling_rate = 8000
high_sampling_rate = 48000
max_target_sr = 48000

val_scale = 3
max_norm = 1
limit = 10240
clip_size_sec = 0.6
batch_size = 8
update_step = 4
scale_res = 50

lr = 5e-4
epochs = 60
step_size = 10
wdc = 1e-3
thershold = 0.8

mdim = 128
ckconv_window = 2.0
encoder_dim = 32
filters = 28

do_adversarial = True
l1_weight = 75
adv_weight = 1
mssl_weight = 10
ema_wt = 0.8
