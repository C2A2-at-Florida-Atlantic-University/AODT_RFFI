import os
import random
import numpy as np
import tensorflow as tf

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Lambda,
    ReLU,
    Add,
    Dense,
    Conv2D,
    Flatten,
    AveragePooling2D,
    Dropout,
    BatchNormalization,
    Reshape,
    Permute,
    LayerNormalization,
    Bidirectional,
    GRU,
)
from tensorflow.keras import initializers

def resblock(x, kernelsize, filters, first_layer=False, seed=None):
    kernel_init = initializers.glorot_uniform(seed=seed)
    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same', kernel_initializer=kernel_init)(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same', kernel_initializer=kernel_init)(fx)
        
        x = Conv2D(filters, 1, padding='same', kernel_initializer=kernel_init)(x)
        
        out = Add()([x, fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same', kernel_initializer=kernel_init)(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same', kernel_initializer=kernel_init)(fx)
              
        out = Add()([x, fx])
        out = ReLU()(out)

    return out 

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)           

class TripletNet():
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
    def create_net(self, embedding_net, alpha):
        self.alpha = alpha
        
        input_shape = [self.datashape[1], self.datashape[2], self.datashape[3]]
        input_1 = Input(input_shape)
        input_2 = Input(input_shape)
        input_3 = Input(input_shape)
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N = embedding_net(input_3)
   
        loss = Lambda(self.triplet_loss)([A, P, N]) 
        model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
        return model
      
    def triplet_loss(self, x):
        # Triplet Loss function.
        anchor, positive, negative = x
        # Distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        # Distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        basic_loss = pos_dist - neg_dist + self.alpha
        loss = K.maximum(basic_loss, 0.0)
        return loss

    def feature_extractor(self, datashape):
        self.datashape = datashape
        input_shape = [self.datashape[1], self.datashape[2], self.datashape[3]]
        inputs = Input(shape=input_shape)
        
        kernel_init = initializers.glorot_uniform(seed=seed_value)
        x = Conv2D(32, 7, strides=2, activation='relu', padding='same', kernel_initializer=kernel_init)(inputs)
        
        x = resblock(x, 3, 32, seed=seed_value)
        x = resblock(x, 3, 32, seed=seed_value)

        x = resblock(x, 3, 64, first_layer=True, seed=seed_value)
        x = resblock(x, 3, 64, seed=seed_value)

        x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)
    
        x = Dense(512, kernel_initializer=kernel_init)(x)
  
        outputs = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model             

    def get_triplet(self):
        a_label = self.rng.choice(self.dev_range)
        n_label = a_label
        while n_label == a_label:
            n_label = self.rng.choice(self.dev_range)
        a = self.call_sample(a_label)
        p = self.call_sample(a_label)
        n = self.call_sample(n_label)
        return a, p, n

    def call_sample(self, label_name):
        indices = np.where(self.label == label_name)[0]
        idx = self.rng.choice(indices)
        return self.data[idx]

    def create_generator(self, batchsize, dev_range, data, label):
        """Generate a triplets generator for training."""
        self.data = data
        self.label = label
        # Use labels present in this split to avoid empty-class sampling.
        self.dev_range = np.unique(self.label)
        if len(self.dev_range) < 2:
            raise ValueError("Triplet training requires at least 2 labels in the current split.")
        
        while True:
            list_a = []
            list_p = []
            list_n = []

            for _ in range(batchsize):
                a, p, n = self.get_triplet()
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)
            
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')
            
            # A "dummy" label which will come into our identity loss
            # function below as y_true. We'll ignore it.
            label = np.ones(batchsize)

            # Keras 3 / TF 2.16 expects tuple-structured inputs for
            # generator -> tf.data conversion (lists trigger TypeSpec errors).
            yield (A, P, N), label


class RNNTripletNet(TripletNet):
    """TripletNet with an RNN encoder over spectrogram time bins."""

    def __init__(
        self,
        seed=42,
        gru_units=256,
        dropout=0.3,
        recurrent_dropout=0.0,
        bidirectional=True,
        num_layers=2,
        embedding_dim=512,
    ):
        super().__init__(seed=seed)
        self.gru_units = gru_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

    def feature_extractor(self, datashape):
        self.datashape = datashape
        f_bins, t_steps, channels = self.datashape[1], self.datashape[2], self.datashape[3]
        inputs = Input(shape=(f_bins, t_steps, channels))

        # Normalize then use a light conv stem before sequence modeling.
        x = LayerNormalization(axis=[1, 2, 3], epsilon=1e-6, name="rnn_triplet_input_norm")(inputs)
        x = Conv2D(64, kernel_size=(3, 3), strides=(2, 1), padding="same", use_bias=False, name="rnn_triplet_conv1")(x)
        x = BatchNormalization(name="rnn_triplet_conv1_bn")(x)
        x = ReLU(name="rnn_triplet_conv1_relu")(x)
        x = Dropout(self.dropout * 0.5, name="rnn_triplet_conv1_dropout")(x)
        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False, name="rnn_triplet_conv2")(x)
        x = BatchNormalization(name="rnn_triplet_conv2_bn")(x)
        x = ReLU(name="rnn_triplet_conv2_relu")(x)

        # Convert conv feature map to sequence: (time, features_per_timestep).
        f_red = int(x.shape[1])
        t_red = int(x.shape[2])
        c_red = int(x.shape[3])
        x = Permute((2, 1, 3))(x)
        x = Reshape((t_red, f_red * c_red))(x)
        x = LayerNormalization(axis=-1, epsilon=1e-6, name="rnn_triplet_step_norm")(x)

        for i in range(self.num_layers):
            return_sequences = i < (self.num_layers - 1)
            gru = GRU(
                self.gru_units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                reset_after=True,
                name=f"rnn_triplet_gru_{i+1}",
            )
            if self.bidirectional:
                x = Bidirectional(gru, name=f"rnn_triplet_bi_gru_{i+1}")(x)
            else:
                x = gru(x)
            if return_sequences:
                x = LayerNormalization(name=f"rnn_triplet_gru_ln_{i+1}")(x)

        hidden_dim = self.gru_units * 2 if self.bidirectional else self.gru_units
        x = Dense(hidden_dim, activation="relu", name="rnn_triplet_projection")(x)
        x = Dropout(self.dropout, name="rnn_triplet_projection_dropout")(x)
        x = Dense(self.embedding_dim, name="rnn_triplet_embedding")(x)
        outputs = Lambda(lambda t: K.l2_normalize(t, axis=1), name="rnn_triplet_l2norm")(x)

        model = Model(inputs=inputs, outputs=outputs, name="RNN_Triplet_Encoder")
        return model


class QuadrupletNet(TripletNet):
    def __init__(self, seed=42):
        super().__init__(seed=seed)
        
    def create_net(self, embedding_net, alpha1, alpha2):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        input_shape = [self.datashape[1], self.datashape[2], self.datashape[3]]
        input_1 = Input(input_shape)
        input_2 = Input(input_shape)
        input_3 = Input(input_shape)
        input_4 = Input(input_shape)
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N1 = embedding_net(input_3)
        N2 = embedding_net(input_4)
   
        loss = Lambda(self.quadruplet_loss)([A, P, N1, N2]) 
        model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=loss)
        return model

    def quadruplet_loss(self, x):
        anchor, positive, negative1, negative2 = x

        # Calculate distances
        ap_dist = K.sum(K.square(anchor - positive), axis=1)
        an1_dist = K.sum(K.square(anchor - negative1), axis=1)
        an2_dist = K.sum(K.square(anchor - negative2), axis=1)
        n1n2_dist = K.sum(K.square(negative1 - negative2), axis=1)

        # Calculate loss
        loss1 = K.maximum(ap_dist - an1_dist + self.alpha1, 0)
        loss2 = K.maximum(ap_dist - an2_dist + self.alpha1, 0)
        loss3 = K.maximum(ap_dist - n1n2_dist + self.alpha2, 0)

        return K.mean(loss1 + loss2 + loss3)

    def get_quadruplet(self):
        """Choose a quadruplet (anchor, positive, negative1, negative2) of images
        such that anchor and positive have the same label and
        negatives have different labels from the anchor."""
        a_label = self.rng.choice(self.dev_range)
        n1_label = a_label
        n2_label = a_label

        while n1_label == a_label:
            n1_label = self.rng.choice(self.dev_range)
        while n2_label == a_label or n2_label == n1_label:
            n2_label = self.rng.choice(self.dev_range)

        a = self.call_sample(a_label)
        p = self.call_sample(a_label)
        n1 = self.call_sample(n1_label)
        n2 = self.call_sample(n2_label)

        return a, p, n1, n2

    def create_generator(self, batchsize, dev_range, data, label):
        """Generate a quadruplets generator for training."""
        self.data = data
        self.label = label
        # Use labels present in this split to avoid empty-class sampling.
        self.dev_range = np.unique(self.label)
        if len(self.dev_range) < 3:
            raise ValueError("Quadruplet training requires at least 3 labels in the current split.")
        
        while True:
            list_a = []
            list_p = []
            list_n1 = []
            list_n2 = []

            for _ in range(batchsize):
                a, p, n1, n2 = self.get_quadruplet()
                list_a.append(a)
                list_p.append(p)
                list_n1.append(n1)
                list_n2.append(n2)
            
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N1 = np.array(list_n1, dtype='float32')
            N2 = np.array(list_n2, dtype='float32')
            
            # A "dummy" label which will come into our identity loss
            # function below as y_true. We'll ignore it.
            label = np.ones(batchsize)

            # Keras 3 / TF 2.16 expects tuple-structured inputs for
            # generator -> tf.data conversion (lists trigger TypeSpec errors).
            yield (A, P, N1, N2), label