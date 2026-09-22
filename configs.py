BASE_PATH = 'Data/VCTKCorpus'
seed = 42
actfe = actfdc = actfdi = actfs = 'relu'
low_sampling_rate = 8000
high_sampling_rate = 48000
max_target_sr = 48000
val_scale = 3
mdim = 128
ckconv_window = 1.5

limit = 10240
clip_size_sec = 0.6
batch_size = 8
update_step = 4
scale_res = 50

lr = 5e-4
epochs = 60
step_size = 10
gamma = 0.2
wdc = 1e-5
thershold = 0.8

max_norm = 1
encoder_dim = 32
filters = 24

do_adversarial = True
optimizer_type = 'adamw' # or 'adabelief'
l1_weight = 80
percp_weight = 10
adv_weight = 1
mssl_weight = 2.5
ema_wt = 0.8