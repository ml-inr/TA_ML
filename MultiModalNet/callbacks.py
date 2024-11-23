import tensorflow as tf

""" Reduce LR on platau
Mods for tracing prediction or uncertanty loss """ 
class custom_LR_platau(tf.keras.callbacks.Callback):

    def __init__(self, block, factor, patience, model, min_delta=0):
        assert block in ['preds','sigmas']
        if block=='preds':
            self.optimizer = model.optimizer_preds
            self.metric = 'val_equal_weighted_summ_pred_loss'
        if block=='sigmas':
            self.optimizer = model.optimizer_sigma
            self.metric = 'val_summ_sigma_loss'
        self.factor = tf.constant(factor, dtype=tf.float32)
        self.patience = tf.constant(patience)
        self.best_metric = tf.constant(1e6)
        self.passed_eps = 0
        self.min_delta = min_delta

    def on_epoch_end(self, epoch, logs):
        if logs[self.metric] + self.min_delta < self.best_metric:
            self.best_metric = logs[self.metric]
            self.passed_eps = 0
        else:
            self.passed_eps = self.passed_eps + 1
            if self.passed_eps >= self.patience:
                self.optimizer.learning_rate.assign( self.optimizer.learning_rate*self.factor )
                self.passed_eps = 0

""" Ensure that metrics are reset """ 
class my_reset_metric(tf.keras.callbacks.Callback):

    def __init__(self, model):
        super().__init__()
        
    def on_epoch_end(self, epoch, logs):
        for metric in self.model.loss_metrics:
            metric.reset_states()