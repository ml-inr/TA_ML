""" Uses non-persistent gradient tape for memory optimization """
import tensorflow as tf

from loss_functions import sigma_loss, cosine_similarity, square_diff_loss

pi = tf.constant(3.141592653, dtype=tf.float32)

# helper function
def get_regul_kwargs(regul_level):
    regul_kwargs = {
            'kernel_regularizer':tf.keras.regularizers.L2(regul_level), 
            'bias_regularizer':tf.keras.regularizers.L2(regul_level)
        }
    return regul_kwargs

""" Network for analyzing detectors as sequance (includes waveform encoder) 
waveforms encodings + detectors parameters -> rnn -> encoding """

# lstm state initializer
class lstm_init_layer(tf.keras.layers.Layer):

    def __init__(self, hidden_size, out_size, regul_kwargs):
        super().__init__()
        self.st_hidd = tf.keras.layers.Dense(hidden_size, **regul_kwargs)
        self.st_out = tf.keras.layers.Dense(out_size+out_size, **regul_kwargs)
        self.activation = activation_function
        self.out_size = out_size
    
    def call(self, inputs, training=False):
        x = self.st_hidd(inputs)
        x = self.activation(x)
        x = self.st_out(x)
        states = self.activation(x)
        return [states[:,:self.out_size], states[:,self.out_size:]]

""" Waveforms encoder
Convs -> rnn -> encodings """

# rnn subnetwork
class wfs_rnn_encoder(tf.keras.layers.Layer):

    def __init__(self, wf_rnn_config):
        super().__init__()
        regul_kwargs = get_regul_kwargs(wf_rnn_config['regul'])
        self.lstm_init_1 = lstm_init_layer(wf_rnn_config['lstm_state_creator_size'], wf_rnn_config['first_lstm_size'], regul_kwargs)
        self.lstm_init_2 = lstm_init_layer(wf_rnn_config['lstm_state_creator_size'], wf_rnn_config['second_lstm_size'], regul_kwargs)
        self.lstm_1 = tf.keras.layers.LSTM(wf_rnn_config['first_lstm_size'], activation='tanh', recurrent_activation='sigmoid', return_sequences=True, **regul_kwargs)
        self.lstm_2 = tf.keras.layers.LSTM(wf_rnn_config['second_lstm_size'], activation='tanh', recurrent_activation='sigmoid', **regul_kwargs)
        self.bn_1 = tf.keras.layers.LayerNormalization()
        self.bn_2 = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs, training=False):
        
        [wf, lstm_init] = inputs
        lstm_state_1 = self.lstm_init_1(lstm_init)
        lstm_state_2 = self.lstm_init_2(lstm_init)

        x = self.lstm_1(wf, initial_state=lstm_state_1)
        x = self.bn_1(x, training=training)
        x = self.lstm_2(x, initial_state=lstm_state_2)
        wf_enc = self.bn_2(x, training=training)

        return wf_enc

# combined waveforms encoder
class wfs_encoder(tf.keras.layers.Layer):

    def __init__(self, wf_config):
        super().__init__()
        regul_kwargs = get_regul_kwargs(wf_config['regul'])
        self.conv_1 = tf.keras.layers.Conv2D(**wf_config['first_2d_conv'], **regul_kwargs)
        self.conv_2 = tf.keras.layers.Conv2D(**wf_config['second_2d_conv'], **regul_kwargs)
        self.conv_3 = tf.keras.layers.Conv1D(**wf_config['last_1d_conv'], **regul_kwargs)
        self.activation = activation_function
        self.bn_1 = tf.keras.layers.LayerNormalization()
        self.bn_2 = tf.keras.layers.LayerNormalization()
        self.bn_3 = tf.keras.layers.LayerNormalization()
        self.dr_1 = tf.keras.layers.Dropout(wf_config['dr_rate'])
        self.dr_2 = tf.keras.layers.Dropout(wf_config['dr_rate'])
        self.dr_3 = tf.keras.layers.Dropout(wf_config['dr_rate'])
        self.rnn_part = wfs_rnn_encoder(wf_config['wf_rrn_params'])
        self.final_encoder = tf.keras.layers.Dense(wf_config['dim_wf_encs'], **regul_kwargs)

    def call(self, inputs, training=False):
        
        [wf, lstm_init] = inputs
        # 2D convs
        x = tf.expand_dims( wf, axis=-1 )
        x = self.conv_1(x)
        x = self.bn_1(x, training=training)
        x = self.dr_1(x, training=training)
        x = self.activation(x)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = self.dr_2(x, training=training)
        x = self.activation(x)

        # 1D conv
        x = tf.squeeze( x, axis=-2 )
        x = self.conv_3(x)
        x = self.bn_3(x, training=training)
        x = self.dr_3(x, training=training)
        x = self.activation(x)

        x = self.rnn_part([x,lstm_init], training=training)
        x = self.final_encoder(x)
        wf_enc = self.activation(x)

        return wf_enc

# analyzing detectors as sequance
class dts_analyzer(tf.keras.layers.Layer):

    def __init__(self, wf_config, rnn_config):
        super().__init__()
        self.wf_encoder = wfs_encoder(wf_config)
        regul_kwargs = get_regul_kwargs(rnn_config['regul'])       
        lstm_layer = tf.keras.layers.LSTM(rnn_config['lstm_units'], activation='tanh', recurrent_activation='sigmoid', **regul_kwargs)
        self.bidir_lstm = tf.keras.layers.Bidirectional( lstm_layer, merge_mode='mul' )
        self.lstm_init_state_creator = lstm_init_layer(rnn_config['lstm_state_creator_size'], rnn_config['lstm_units'], regul_kwargs)
        self.final_encoder = tf.keras.layers.Dense(rnn_config['dim_det_enc'], **regul_kwargs)
        self.activation = activation_function
        self.dim_wf_encs = wf_config['dim_wf_encs']
    
    def call(self, inputs, training=False):
        [wf_fl, mask_fl, dt_fl, lstm_init] = inputs

        wf_resh = tf.reshape(wf_fl, (-1,128,2))
        mask_resh = tf.reshape(mask_fl, (-1,1))
        lstm_init_resh = tf.reshape(lstm_init, (-1,10))
        wf_encs = tf.where(mask_resh, self.wf_encoder([wf_resh,lstm_init_resh], training=training), tf.zeros(self.dim_wf_encs) )
        wf_encs = tf.reshape(wf_encs, (-1,tf.shape(dt_fl)[1], self.dim_wf_encs))

        wf_dt = tf.concat([wf_encs,dt_fl], axis=-1)

        ltsm_init_full = lstm_init[:,0,:5]
        lstm_st = self.lstm_init_state_creator(ltsm_init_full)

        # due to presence of mask in call, many warnings. seems ok.
        rnn_res = self.bidir_lstm(inputs=wf_dt, initial_state=[lstm_st[0],lstm_st[1],lstm_st[0],lstm_st[1]], mask=mask_fl)
        x = self.final_encoder(rnn_res)
        wfs_encoded = self.activation(x)
        return wfs_encoded

""" Detectors grid encoder """
class dtbundle_encoder(tf.keras.layers.Layer):

    def __init__(self, config):
        super().__init__()
        regul_kwargs = get_regul_kwargs(config['regul'])  
        self.conv_1 = tf.keras.layers.Conv2D(**config['spatil_1_conv'], **regul_kwargs)
        self.conv_2 = tf.keras.layers.Conv2D(**config['spatil_2_conv'], **regul_kwargs)
        self.conv_3 = tf.keras.layers.Conv2D(**config['spatil_3_conv'], **regul_kwargs)
        self.conv_4 = tf.keras.layers.Conv2D(**config['spatil_4_conv'], **regul_kwargs)
        self.pool_1 = tf.keras.layers.AveragePooling2D(**config['spatil_pool'])
        self.activation = activation_function
        self.bn_1 = tf.keras.layers.LayerNormalization()
        self.bn_2 = tf.keras.layers.LayerNormalization()
        self.bn_3 = tf.keras.layers.LayerNormalization()
        self.bn_4 = tf.keras.layers.LayerNormalization()
        self.dr_1 = tf.keras.layers.Dropout(config['dr_rate'])
        self.dr_2 = tf.keras.layers.Dropout(config['dr_rate'])
        self.dr_3 = tf.keras.layers.Dropout(config['dr_rate'])
        self.dr_4 = tf.keras.layers.Dropout(config['dr_rate'])
        self.flatten = tf.keras.layers.Flatten()
        self.final_encoder = tf.keras.layers.Dense(config['dim_grid_encs'], **regul_kwargs)
    
    def call(self, inputs, training=False):
        
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = self.dr_1(x, training=training)
        x = self.activation(x)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = self.dr_2(x, training=training)
        x = self.activation(x)
        x = self.pool_1(x)

        x = self.conv_3(x)
        x = self.bn_3(x, training=training)
        x = self.dr_3(x, training=training)
        x = self.activation(x)

        x = self.conv_4(x)
        x = self.bn_4(x, training=training)
        x = self.dr_4(x, training=training)
        x = self.activation(x)

        x = self.flatten(x)
        enc = self.final_encoder(x, training=training)
        return enc

""" Predictions from encodings
outputs (mass, logE, 3D direction vector) """
class union_analyzer(tf.keras.layers.Layer):

    def __init__(self, config_dict):
        super().__init__()
        regul_kwargs = get_regul_kwargs(config_dict['regularization'])
        self.denses = [ tf.keras.layers.Dense(u, **regul_kwargs) for u in config_dict['units'] ]
        self.drops = [ tf.keras.layers.Dropout(d) for d in config_dict['drops'] ]
        self.bns = [ tf.keras.layers.LayerNormalization() for _ in config_dict['units'] ]
        
        self.activation = activation_function
        self.me_layer = tf.keras.layers.Dense(2, **regul_kwargs)
        self.dir_layer = tf.keras.layers.Dense(3, use_bias=False, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), **regul_kwargs)
        
    def call(self, inputs, training=False):
        
        x = inputs
        for [dense,drop,bn] in zip(self.denses, self.drops, self.bns):
            x = dense(x)
            x = drop(x, training=training)
            x = self.activation(x)
            x = bn(x, training=training)
        me = self.me_layer(x)
        dir = self.dir_layer(x)
        return me, dir

""" Uncertanty estimation
outputs 3 sigmas, for (mass, logE, direction) """
class sigma_predictor(tf.keras.layers.Layer):

    def __init__(self, config_dict):
        super().__init__()
        regul_kwargs = get_regul_kwargs(config_dict['regularization'])
        self.denses = [ tf.keras.layers.Dense(u, **regul_kwargs) for u in config_dict['units'] ]
        self.drops = [ tf.keras.layers.Dropout(d) for d in config_dict['drops'] ]
        self.bns = [ tf.keras.layers.BatchNormalization() for _ in config_dict['units'] ]
        self.activation = activation_function

        self.last_activation = tf.keras.activations.softplus
        self.last_layer = tf.keras.layers.Dense(3, **regul_kwargs)
    
    def call(self, inputs, training=False):
        
        x = inputs
        for [dense,drop,bn] in zip(self.denses, self.drops, self.bns):
            x = dense(x)
            x = drop(x, training=training)
            x = self.activation(x)
            x = bn(x, training=training)
        x = self.last_layer(x)
        sigmas = self.last_activation(x)
        return sigmas

""" Encoder combining features from varios data representaions """
class encoder(tf.keras.Model):

    def __init__(self, grid_config, wf_config, rnn_config):
        super().__init__()
        self.dtbundle_encoder = dtbundle_encoder(grid_config)
        self.dts_encoder = dts_analyzer(wf_config, rnn_config)
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=False):
        
        [dt_fl_in, wfs_in, gp_in, dt_in] = inputs
        mask_wfs_in = tf.cast( dt_fl_in[:,:,-1], bool )

        dt_enc = self.dtbundle_encoder(dt_in, training=training)
            
        gp_to_lstm = tf.expand_dims( tf.concat( (gp_in[:,0:3],gp_in[:,6:7],gp_in[:,9:10]), axis=-1), axis=1)
        gp_to_lstm = tf.repeat(gp_to_lstm, repeats=tf.shape(dt_fl_in)[1], axis=1)
        lstm_init = tf.concat([dt_fl_in[:,:,0:3], dt_fl_in[:,:,4:6], gp_to_lstm], axis=-1 )
        wfs_enc = self.dts_encoder([wfs_in,mask_wfs_in,dt_fl_in[:,:,:-1],lstm_init], training=training)

        un_enc = self.concat([gp_in,dt_enc,wfs_enc])
        return un_enc

""" Full model """   
class TA_net(tf.keras.Model):

    def __init__(self, config_dict):
        super().__init__()
        self.encoder = encoder(config_dict['dt_geom_grid'], config_dict['wf_encoder_params'], config_dict['dt_sequence_analyzer'])
        self.union_analyzer = union_analyzer(config_dict['preds_branch_params'])
        self.sigma_predictor = sigma_predictor(config_dict['sigmas_branch_params'])
        self.init_config_dict = config_dict

    def compile(self, config_dict):
        super().compile()
        # init optimizers
        optimizer_preds_config = config_dict['optimizers']['optimizer_preds']
        self.optimizer_preds = getattr(tf.keras.optimizers, optimizer_preds_config['name'])(**optimizer_preds_config['kwargs'])
        optimizer_sigma_config = config_dict['optimizers']['optimizer_sigma']
        self.optimizer_sigma = getattr(tf.keras.optimizers, optimizer_sigma_config['name'])(**optimizer_sigma_config['kwargs'])
        # init weights
        weights_dict = config_dict['weights_training_parameters']
        self.loss_weights_preds = tf.constant(weights_dict['weights_loss_preds'], dtype=tf.float32)
        self.loss_weights_sigma = tf.constant(weights_dict['weights_loss_sigma'], dtype=tf.float32)
        self.loss_weights_calls = tf.constant(weights_dict['weights_loss_calls'], dtype=tf.float32)
        self.adds_loss_calls = tf.constant([0., 0., 1.], dtype=tf.float32)
        # init loss function
        self.sigma_loss_fn = sigma_loss() 
        self.square_diff_loss = square_diff_loss()
        self.cosine_similarity = cosine_similarity()
        # init metrics
        loss_names = ['lnA_mse_loss', 'lnE_mse_loss', 'dir_sim_loss', 'summ_prediction_loss',
                        'lnA_sigma_loss', 'lnE_sigma_loss', 'dir_sim_sigma_loss', 'summ_sigma_loss',
                        'equal_weighted_summ_pred_loss']
        self.loss_metrics = [ tf.keras.metrics.Mean(name=mn) for mn in loss_names ]
        # sigma prediction normalization
        self.means_preds = tf.constant(config_dict['means_preds'], dtype=tf.float32)

    def build(self, input_shape):
        super().build([input_shape[0], input_shape[1], input_shape[2], input_shape[3]])
        self.built = True
        # store the variable counts
        self.n_pred_vars = len(self.encoder.trainable_variables + self.union_analyzer.trainable_variables)
        self.n_sigma_vars = len(self.sigma_predictor.trainable_variables)

    @property
    def metrics(self):
        return self.loss_metrics

    def get_config(self):
        config = super().get_config()
        config.update(self.init_config_dict)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # use float 64 for better numerical stability
    def cast_to_dir_vector_64(self, spherical):
        spherical = tf.cast(spherical, tf.float64)
        x = tf.math.sin(spherical[:,0]) * tf.math.cos(spherical[:,1])
        y = tf.math.sin(spherical[:,0]) * tf.math.sin(spherical[:,1])
        z = tf.math.cos(spherical[:,0])
        return tf.stack([x, y, z], axis=-1)

    # evaluate prediction loss
    def eval_pred_loss(self, preds, labels, weights):
        el_lnA_loss = self.square_diff_loss(labels[:,0], preds[:,0], weights)
        el_lnE_loss = self.square_diff_loss(labels[:,1], preds[:,1], weights)
        el_dir_loss = tf.cast(self.cosine_similarity(preds[:,2:5], self.cast_to_dir_vector_64(labels[:,2:4]), weights), tf.float32)
        lnA_mse_loss = tf.math.reduce_mean(el_lnA_loss)
        lnE_mse_loss = tf.math.reduce_mean(el_lnE_loss)
        dir_tot_loss = tf.math.reduce_mean(el_dir_loss) 
        # metric should come in that specific order
        return [lnA_mse_loss, lnE_mse_loss, dir_tot_loss], [el_lnA_loss, el_lnE_loss, el_dir_loss]

    # evaluatr uncertanty loss
    def eval_sigma_loss(self, element_losses, sigmas):
        lnA_sigma_loss = tf.math.reduce_mean(self.sigma_loss_fn( element_losses[0]/self.means_preds[0], sigmas[:,0] ))
        lnE_sigma_loss = tf.math.reduce_mean(self.sigma_loss_fn( element_losses[1]/self.means_preds[1], sigmas[:,1] ))
        dir_sigma_loss = tf.math.reduce_mean(self.sigma_loss_fn( (1.+element_losses[2])/self.means_preds[2], sigmas[:,2] ))
        # metric should come in that specific order
        return [lnA_sigma_loss, lnE_sigma_loss, dir_sigma_loss]

    def update_custom_metrics(self, metric_updates):
        metric_updates[2] = 1. + metric_updates[2]
        for m_update, m_tracker in zip(metric_updates, self.loss_metrics):
            m_tracker.update_state(m_update)
        
    def call(self, inputs, training=False):
        
        [dt_fl_in, wfs_in, gp_in, dt_in] = inputs
        encs = self.encoder([dt_fl_in, wfs_in, gp_in, dt_in], training=training)
        
        mass_energy, direction = self.union_analyzer(encs, training=training) # me = mass, energy
        # make sure sigma predictor does not influence prediction branch
        extnd = tf.concat([encs, mass_energy, direction], axis=-1)
        extnd_stoped = tf.stop_gradient(extnd)
        sigmas = self.sigma_predictor(extnd_stoped, training=training)

        out_preds = tf.concat([mass_energy, direction], axis=-1)
        return out_preds, sigmas

    def train_step(self, inputs):
        [data, labels, weights] = inputs
        with tf.GradientTape(persistent=False) as tape: 
            preds, sigmas = self.__call__(data, training=True)
            # calculate loss
            preds_losses, element_losses = self.eval_pred_loss(preds, labels, weights)
            element_losses_stopped = tf.stop_gradient(element_losses)
            # do not allow gradient leak via losses
            sigma_losses = self.eval_sigma_loss(element_losses_stopped, sigmas)
            res_pred_loss = tf.math.reduce_sum(self.loss_weights_preds*tf.stack(preds_losses, axis=0))
            res_sigma_loss = tf.math.reduce_sum(self.loss_weights_sigma*tf.stack(sigma_losses, axis=0))
            res_loss = res_pred_loss + res_sigma_loss
        
        # normilized loss for callbacks
        calls_loss = tf.math.reduce_sum(self.loss_weights_calls*(self.adds_loss_calls+tf.stack(preds_losses, axis=0)))
        # calculate gradients
        grads = tape.gradient(res_loss, self.encoder.trainable_variables + self.union_analyzer.trainable_variables + self.sigma_predictor.trainable_variables)      
        pred_grads = grads[:self.n_pred_vars]
        sigma_grads = grads[-self.n_sigma_vars:]
        self.optimizer_preds.apply_gradients(zip(pred_grads, self.encoder.trainable_variables + self.union_analyzer.trainable_variables))
        self.optimizer_sigma.apply_gradients(zip(sigma_grads, self.sigma_predictor.trainable_variables))
        # update metrics
        self.update_custom_metrics(preds_losses+[res_pred_loss]+sigma_losses+[res_sigma_loss]+[calls_loss])
        return {**{m_tracker.name : m_tracker.result() for m_tracker in self.loss_metrics},
               "learning_rate_preds":self.optimizer_preds.learning_rate,
                "learning_rate_sigma":self.optimizer_sigma.learning_rate }
   
    def test_step(self, inputs):
        [data, labels, weights] = inputs
        # predict
        preds, sigmas = self.__call__(data, training=False)
        # calculate loss
        preds_losses, element_losses = self.eval_pred_loss(preds, labels, weights)
        sigma_losses = self.eval_sigma_loss(element_losses, sigmas)
        res_pred_loss = tf.math.reduce_sum(self.loss_weights_preds*tf.stack(preds_losses, axis=0))
        res_sigma_loss = tf.math.reduce_sum(self.loss_weights_sigma*tf.stack(sigma_losses, axis=0))
        calls_loss = tf.math.reduce_sum(self.loss_weights_calls*(self.adds_loss_calls+tf.stack(preds_losses, axis=0)))
        # update metrics
        self.update_custom_metrics(preds_losses+[res_pred_loss]+sigma_losses+[res_sigma_loss]+[calls_loss])
        return {**{m_tracker.name : m_tracker.result() for m_tracker in self.loss_metrics },
               "learning_rate_preds":self.optimizer_preds.learning_rate,
                "learning_rate_sigma":self.optimizer_sigma.learning_rate }

def make_TA_net(nn_config_dict):
    global activation_function
    activation_function = getattr(tf.keras.activations, nn_config_dict['nn_act_function'])
    return TA_net(nn_config_dict)