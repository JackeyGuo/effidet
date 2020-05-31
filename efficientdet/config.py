compound_coef = 2
image_size = None
num_workers = 8
batch_size = 5

# Optimizer parameters
opt = 'adamw'
opt_eps = 1e-8
momentum = 0.9
weight_decay = 0.0001

sched = 'step'
patience_epochs = 10
lr = 1e-3
min_lr = 1e-7
warmup_lr = 0.0001
num_epochs = 100
start_epoch = 0
decay_epochs = 10
warmup_epochs = 3
cooldown_epochs = 10
decay_rate = 0.1

val_interval = 1
es_min_delta = 0.0
date = 'd2-0530'
log_path = 'logs/%s' % date
saved_path = 'outputs/%s' % date
