import tensorflow as tf
import numpy as np
import h5py as h5

# conversion constant
deg2rad = 3.14159265/180

""" Generator reading batches of data from hdf5 file.
    Applies data augmentation and yields ragged TF tensors"""
class batch_data_generator:
    
    def __init__(self, file, regime, 
                 batch_size, weights, return_reminder,
                 norm_E_params,
                 apply_add_gauss, gauss_stds,
                 def_vals_dt, def_vals_wfs,
                 apply_mult_gauss, mult_gauss_std):
        
        # init main parameters
        self.file = file
        self.regime = regime
        self.batch_size = batch_size
        self.weights = np.array(weights).astype(np.float32)
        self.norm_E_params = np.array(norm_E_params).astype(np.float32)
        self.apply_add_gauss = apply_add_gauss
        self.apply_mult_gauss = apply_mult_gauss
        self.g_mult_stds = mult_gauss_std
        self.def_vals_dt = np.array(def_vals_dt).astype(np.float32)
        self.def_vals_wfs = np.array(def_vals_wfs).astype(np.float32)
        self.hf = h5.File(self.file,'r')
        self.dsets = ['dt_params', 'wfs_flat', 'recos', 'dt_bundle']

        # calculate the number of steps
        self.num = self.hf[self.regime+'/ev_starts'].shape[0]-1
        self.batch_num = self.num // self.batch_size
        if return_reminder and (self.num % self.batch_size)!=0:
            self.batch_num += 1

        # init noise generation parameters
        self.mult_const = {}
        self.g_add_stds = {}
        for ds in self.dsets:
            mean = self.hf['norm_param/'+ds+'/mean'][()]
            std = self.hf['norm_param/'+ds+'/std'][()]
            if self.apply_add_gauss[ds]:
                self.g_add_stds[ds] = gauss_stds[ds]/std
            if ds=='wfs_flat' and self.apply_mult_gauss['wfs_flat']:
                self.mult_const[ds] = mean/std
            if ds=='recos' and self.apply_mult_gauss['recos']:
                self.mult_const[ds] = mean[11]/std[11]
            if ds=='dt_params' and self.apply_mult_gauss['dt_params']:
                self.mult_const[ds] = mean[3]/std[3]
            if ds=='dt_bundle' and self.apply_mult_gauss['dt_bundle']:
                self.mult_const[ds] = mean[3]/std[3]

    # additive gaussian noise
    def add_gauss(self, data, std):
        g_add_stds = np.broadcast_to(std, data.shape)
        noise = np.random.normal(scale=g_add_stds, size=data.shape).astype(np.float32)
        data = data + noise
        return data

    # multiplicative gaussian noise
    def mult_gauss(self, data, std, norm_const):
        g_mult_stds = np.broadcast_to(std, data.shape)
        noise = np.random.normal(scale=g_mult_stds, size=data.shape)
        data += noise*(data + norm_const)
        return data

    # data reading and preparation step
    def step(self, start_ev, stop_ev, start_det, stop_det):
        # read data
        dt_params = self.hf[self.regime+'/dt_params'][start_det:stop_det]
        wfs_flat = self.hf[self.regime+'/wfs_flat'][start_det:stop_det]
        recos = self.hf[self.regime+'/recos'][start_ev:stop_ev]
        dt_bundle = self.hf[self.regime+'/'+'dt_bundle'][start_ev:stop_ev]
        mask = np.ones( (dt_params.shape[:-1]+(1,)), dtype=np.float32 )
        # make labels
        mc_corsica_partilce = self.hf[self.regime+'/mc_params/'][start_ev:stop_ev,1:2]
        particle_labels = np.where( mc_corsica_partilce==14, -1., 1. )
        energy_labels = np.log10(self.hf[self.regime+'/mc_params/'][start_ev:stop_ev,3:4])
        energy_labels = (energy_labels - self.norm_E_params[0]) / self.norm_E_params[1]
        angles_labels = self.hf[self.regime+'/mc_params/'][start_ev:stop_ev,4:6]*deg2rad
        ws = np.where( mc_corsica_partilce[:,0]==14, self.weights[0], self.weights[1] )
        labels = np.concatenate( (particle_labels, energy_labels, angles_labels), axis=-1 )
        # apply additive noise
        if self.apply_add_gauss['dt_params']:
            dt_params = self.add_gauss(dt_params, self.g_add_stds['dt_params'])
        if self.apply_add_gauss['wfs_flat']:
            wfs_flat = self.add_gauss(wfs_flat, self.g_add_stds['wfs_flat'])
        if self.apply_add_gauss['recos']:
            recos = self.add_gauss(recos, self.g_add_stds['recos'])
        if self.apply_add_gauss['dt_bundle']:
            dt_bundle = self.add_gauss(dt_bundle, self.g_add_stds['dt_bundle'])
        # apply multipllicative noise
        if self.apply_mult_gauss['dt_params']:
            dt_params[...,3] = self.mult_gauss(dt_params[...,3], self.g_mult_stds['dt_params'], self.mult_const['dt_params'])
        if self.apply_mult_gauss['wfs_flat']:
            wfs_flat = self.mult_gauss(wfs_flat, self.g_mult_stds['wfs_flat'], self.mult_const['wfs_flat'])
        if self.apply_mult_gauss['recos']:
            recos[...,11] = self.mult_gauss(recos[...,11], self.g_mult_stds['recos'], self.mult_const['recos'])
        if self.apply_mult_gauss['dt_bundle']:
            dt_bundle[...,3] = self.mult_gauss(dt_bundle[...,3], self.g_mult_stds['dt_bundle'], self.mult_const['dt_bundle'])
        dt_params = np.concatenate( (dt_params,mask), axis=-1 )
        return ((dt_params,wfs_flat,recos,dt_bundle), labels, ws)

    # read, preprocess data and yield ragged TF tensors
    def __call__(self):
        start_ev = 0
        stop_ev = self.batch_size
        for i in range(self.batch_num):
            # read and preprocess data
            ev_idxs = self.hf[self.regime+'/ev_starts'][start_ev:stop_ev+1]
            ( (dt_params, wfs_flat, recos, dt_bundle), labels, ws ) = self.step(start_ev, stop_ev, ev_idxs[0], ev_idxs[-1] )
            # make ragged tensors
            raw_length = np.diff(ev_idxs)
            dt_params = tf.RaggedTensor.from_row_lengths( dt_params, raw_length, validate=False )
            dt_params = dt_params.to_tensor( default_value=self.def_vals_dt )
            wfs_flat = tf.cast( tf.RaggedTensor.from_row_lengths( wfs_flat, raw_length, validate=False ), tf.float32 )
            wfs_flat = wfs_flat.to_tensor( default_value=self.def_vals_wfs )
            start_ev += self.batch_size
            stop_ev += self.batch_size
            yield ( ( dt_params, wfs_flat, recos, dt_bundle ), labels, ws )

""" Makes TF dataset"""
def make_datasets(config_dict):
    # init variables
    for key, value in config_dict.items():
        globals()[key] = value
    if isinstance(prefetch_method, int):
        how_to_prefetch = prefetch_method
    else:
        how_to_prefetch = tf.data.AUTOTUNE
    
    if return_reminder:
        train_batch_size = None
    else:
        train_batch_size = batch_size

    # generator for training data
    tr_generator = batch_data_generator(h5f_train, 'train', 
                                        train_batch_size, weights_particles, return_reminder,
                                        norm_E_params,
                                        apply_gauss, gauss_stds, 
                                        dense_def_vals_dts, dense_def_vals_wfs, 
                                        apply_mult_gauss, gauss_mult_stds)

    output_signature = (
        (
         tf.TensorSpec(shape=(None, None, 7)), 
         tf.TensorSpec(shape=(None, None, 128, 2)),
         tf.TensorSpec(shape=(None, 15)), 
         tf.TensorSpec(shape=(None, 6, 6, 7))
         ),
        tf.TensorSpec(shape=(None, 4)),
        tf.TensorSpec(shape=(None,))
    )
    
    train_dataset = tf.data.Dataset.from_generator( tr_generator, 
                        output_signature=output_signature ).repeat(-1).prefetch(how_to_prefetch)

    # generator for test data
    apply_gauss_test = {'dt_params':False, 'wfs_flat':False, 'recos':False, 'dt_bundle':False}

    te_generator = batch_data_generator(h5f_train, 'test', 
                                        train_batch_size, weights_particles, return_reminder,
                                        norm_E_params,
                                        apply_gauss_test, None, 
                                        dense_def_vals_dts, dense_def_vals_wfs, 
                                        apply_gauss_test, None)
    
    test_dataset = tf.data.Dataset.from_generator( te_generator, 
                        output_signature=output_signature ).prefetch(how_to_prefetch)
    
    return train_dataset, test_dataset