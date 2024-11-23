import tensorflow as tf
import numpy as np
import h5py as h5
import yaml
import shutil

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from nn_arch import make_TA_net
from data_generator import make_datasets
from callbacks import custom_LR_platau, my_reset_metric

# read YAML
with open('config_train.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
# and config globals
for key, value in config['util_config'].items():
    globals()[key] = value

# make datasets
train_ds, test_ds = make_datasets(config['dataset_params'])

# save config
shutil.copy('config_train.yaml', f"{save_path}{model_name}.yaml.config")

# phase 1: train prediction branch
print("Phase 1: Training prediction branch...")
classifier = make_TA_net(config['nn_arch_config'])
classifier.sigma_predictor.trainable = False  # freeze sigma predictor
classifier.compile(config['nn_compile_config'])

TB_cb = tf.keras.callbacks.TensorBoard(log_dir=log_path+model_name+"_phase1")
ES_cb = tf.keras.callbacks.EarlyStopping(monitor='val_equal_weighted_summ_pred_loss', min_delta=min_delta_early_stop, 
                                       patience=patience_early_stop, restore_best_weights=True, mode='min', verbose=1)
CP_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_path+model_name+'_phase1_ckpt', 
                                         monitor='val_equal_weighted_summ_pred_loss', save_best_only=True, 
                                         save_weights_only=False, save_freq='epoch')
reset_metric = my_reset_metric(classifier)
cb_preds = custom_LR_platau('preds', plateau_lr_factor, plateau_lr_patience, classifier)

classifier.fit(train_ds, validation_data=test_ds, epochs=100, verbose=0, 
              callbacks=[TB_cb, ES_cb, CP_cb, reset_metric, cb_preds],
              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
classifier.save(save_path+model_name+'_phase1')

# phase 2: train sigma predictor
print("Phase 2: Training sigma predictor...")
classifier.sigma_predictor.trainable = True
classifier.encoder.trainable = False
classifier.union_analyzer.trainable = False

TB_cb = tf.keras.callbacks.TensorBoard(log_dir=log_path+model_name+"_phase2")
ES_cb = tf.keras.callbacks.EarlyStopping(monitor='val_summ_sigma_loss', min_delta=min_delta_early_stop_sigma, 
                                       patience=patience_early_stop_sigma, restore_best_weights=True, mode='min', verbose=1)
CP_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_path+model_name+'_phase2_ckpt', 
                                         monitor='val_summ_sigma_loss', save_best_only=True, 
                                         save_weights_only=False, save_freq='epoch')
cb_sigmas = custom_LR_platau('sigmas', plateau_lr_factor_sigma, plateau_lr_patience_sigma, classifier)

classifier.compile(config['nn_compile_config'])  # recompile after changing trainable states
classifier.fit(train_ds, validation_data=test_ds, epochs=100, verbose=0,
              callbacks=[TB_cb, ES_cb, CP_cb, reset_metric, cb_sigmas],
              steps_per_epoch=steps_per_epoch_sigma, validation_steps=validation_steps_sigma)

classifier.save(save_path+model_name)