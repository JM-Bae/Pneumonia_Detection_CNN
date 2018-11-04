# Loss Functions 
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.)/(tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)
                                 - intersection +1.)
    return 1 - score

def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + \
           0.5 * iou_loss(y_true, y_pred)

def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_pred * y_true, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) +\
            tf.reduce_sum(y_pred, axis=[1,2,3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect+smooth)/(union-intersect+smooth))

# define learning rate
def cosine_annealing(x):
    lr = 0.001
    epochs = 25
    return lr*(np.cos(np.pi*x/epochs)+1.)/2