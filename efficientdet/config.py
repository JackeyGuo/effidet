compound_coef = 2
image_size = None
num_workers = 8
batch_size = 8

# Optimizer parameters
opt = 'lookahead_radam'
opt_eps = 1e-8
momentum = 0.9
weight_decay = 0.0001
sched = 'step'
lr = 0.001
warmup_lr = 0.0001
num_epochs = 200
start_epoch = 0
decay_epochs = 40
warmup_epochs = 3
decay_rate = 0.1

val_interval = 1
es_min_delta = 0.0
date = 'd1-0528'
log_path = 'logs/%s' % date
saved_path = 'outputs/%s' % date
