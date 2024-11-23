import tensorflow as tf

""" Cosine similarity loss with float64 option
Range [-1,1] """
class cosine_similarity(tf.keras.losses.Loss):

    def __init__(self, use_float64 = True):
        super().__init__()
        self.use_float64 = use_float64

    def get_config(self):
        config = super().get_config()
        config.update({"use_float64" : self.use_float64})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def __call__(self, true_vector, pred_vector, sample_weight=None):
        
        if self.use_float64:
            true_vector = tf.cast(true_vector, tf.float64)
            pred_vector = tf.cast(pred_vector, tf.float64)

        dot_product = tf.reduce_sum(true_vector * pred_vector, axis=-1)
        true_norm = tf.sqrt(tf.reduce_sum(tf.square(true_vector), axis=-1))
        pred_norm = tf.sqrt(tf.reduce_sum(tf.square(pred_vector), axis=-1))
        
        loss = - dot_product / (true_norm * pred_norm)

        if sample_weight is not None:
            sample_weight = tf.squeeze(sample_weight)
            loss = tf.cast(sample_weight, tf.float64)*loss

        return loss

""" Squared residuals """ 
class square_diff_loss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        diff = y_true-y_pred
        loss = diff*diff
        if sample_weight is not None:
            sample_weight = tf.squeeze(sample_weight)
            loss = sample_weight*loss
        return loss

""" Loss for estimating predictions uncertainty """ 
class sigma_loss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()
        
    def __call__(self, pred_loss, sigma_pred, sample_weight=None):
        loss = tf.math.log(sigma_pred+1e-6) + pred_loss/(sigma_pred+1e-6)
        if sample_weight is not None:
            sample_weight = tf.squeeze(sample_weight)
            loss = sample_weight*loss
        return loss