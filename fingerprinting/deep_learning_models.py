import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, ReLU, Add, Dense, Conv2D, Flatten, AveragePooling2D, TimeDistributed

def resblock(x, kernelsize, filters, first_layer = False):

    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        
        x = Conv2D(filters, 1, padding='same')(x)
        
        out = Add()([x,fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same')(fx)
          
        out = Add()([x,fx])
        out = ReLU()(out)

    return out 

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)           

class TripletNet():
    def __init__(self):
        pass
        
    def create_triplet_net(self, embedding_net, alpha, tough_coef):
        
        #        embedding_net = encoder()
        self.alpha = alpha
        self.tough_coef = tough_coef
        
        input_1 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_2 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_3 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N = embedding_net(input_3)
   
        loss = Lambda(self.triplet_loss)([A, P, N]) 
        model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
        return model
      
    def triplet_loss(self,x):
        # Triplet Loss function.
        anchor,positive,negative = x
        #        K.l2_normalize
        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor-positive),axis=1)
        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor-negative),axis=1)

        # Let's make it harder by artificially worsening the distances
        neg_dist = neg_dist * self.tough_coef # lie to the system by making the distance smaller

        basic_loss = pos_dist-neg_dist + self.alpha
        loss = K.maximum(basic_loss,0.0)
        return loss
    
    def feature_extractor(self, datashape):
            
        self.datashape = datashape
        
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)
        
        x = resblock(x, 3, 32)
        x = resblock(x, 3, 32)

        x = resblock(x, 3, 64, first_layer = True)
        x = resblock(x, 3, 64)

        x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)
    
        x = Dense(512)(x)
  
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model             

    
    def get_triplet(self):
        """Choose a triplet (anchor, positive, negative) of images
        such that anchor and positive have the same label and
        anchor and negative have different labels."""
        
        
        n = a = self.dev_range[np.random.randint(len(self.dev_range))]
        
        while n == a:
            # keep searching randomly!
            n = self.dev_range[np.random.randint(len(self.dev_range))]
        a, p = self.call_sample(a), self.call_sample(a)
        n = self.call_sample(n)
        
        return a, p, n

          
    def call_sample(self,label_name):
        """Choose an image from our training or test data with the
        given label."""
        num_sample = len(self.label)
        idx = np.random.randint(num_sample)
        while self.label[idx] != label_name:
            # keep searching randomly!
            idx = np.random.randint(num_sample) 
        return self.data[idx]


    def create_generator(self, batchsize, dev_range, data, label):
        """Generate a triplets generator for training."""
        self.data = data
        self.label = label
        self.dev_range = dev_range
        
        while True:
            list_a = []
            list_p = []
            list_n = []

            for i in range(batchsize):
                a, p, n = self.get_triplet()
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)
            
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')
            
            # a "dummy" label which will come in to our identity loss
            # function below as y_true. We'll ignore it.
            label = np.ones(batchsize)

            yield [A, P, N], label  

class NPairNet():
    def __init__(self):
        pass

    def feature_extractor(self, datashape):
        self.datashape = datashape
        
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)
        x = resblock(x, 3, 32)
        x = resblock(x, 3, 32)
        x = resblock(x, 3, 64, first_layer = True)
        x = resblock(x, 3, 64)
        x = AveragePooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
  
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model 

    def create_npair_net(self, feature_extractor, alpha, num_neg, loss_type):
        self.num_neg = num_neg
        self.alpha = alpha

        # The inputs are: [A, P, N1, ..., NN]
        model_inputs = [Input([self.datashape[1],self.datashape[2],self.datashape[3]]) for _ in np.arange(self.num_neg + 2)]
        loss_inputs = [feature_extractor(input) for input in model_inputs]

        if loss_type == 'triplet_loss':
            model_outputs = Lambda(self.triplet_loss)(loss_inputs)
        elif loss_type == 'n_loss':
            model_outputs = Lambda(self.n_loss)(loss_inputs)
        elif loss_type == 'quadruplet_loss':
            model_outputs = Lambda(self.quadruplet_loss)(loss_inputs)
        elif loss_type == 'quintuplet_loss':
            model_outputs = Lambda(self.quintuplet_loss)(loss_inputs)
        else: raise Exception('Invalid loss function type. Accepted options: [triplet_loss, quadruplet_loss, quintuplet_loss, n_loss]')

        print(f'Applying {loss_type}')

        return Model(inputs = model_inputs, outputs = model_outputs)
        
    def n_loss(self, outputs):
        anchor = outputs[0]
        positive = outputs[1]
        negatives = outputs[2:]

        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dists = [K.sum(K.square(anchor - negative), axis=1) for negative in negatives]

        return K.sum([K.maximum(0.0, pos_dist - neg_dist + self.alpha) for neg_dist in neg_dists])

    def triplet_loss(self, outputs):
        anchor = outputs[0]
        positive = outputs[1]
        negative = outputs[2]

        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        return K.maximum(0.0, pos_dist-neg_dist + self.alpha)

    def quadruplet_loss(self, outputs):
        anchor = outputs[0]
        positive = outputs[1]
        negative1 = outputs[2]
        negative2 = outputs[3]

        ap = K.sum(K.square(anchor - positive), axis=1)
        an = K.sum(K.square(anchor - negative1), axis=1)
        nn = K.sum(K.square(negative1 - negative2), axis=1)

        return K.maximum(0.0, (ap - an + self.alpha) + (ap - nn + self.alpha / 2.0))

    def quintuplet_loss(self, outputs):
        anchor = outputs[0]
        positive = outputs[1]
        negative1 = outputs[2]
        negative2 = outputs[3]
        negative3 = outputs[4]

        ap_dist = K.sum(K.square(anchor - positive), axis=1)
        an1_dist = K.sum(K.square(anchor - negative1), axis=1)
        an2_dist = K.sum(K.square(anchor - negative2), axis=1)
        an3_dist = K.sum(K.square(anchor - negative3), axis=1)

        loss = K.maximum(0.0, ap_dist - an1_dist + self.alpha) + \
            K.maximum(0.0, ap_dist - an2_dist + self.alpha) + \
            K.maximum(0.0, ap_dist - an3_dist + self.alpha)

        return loss
    
    def create_generator(self, batchsize, dev_range, data, label, npair_type):
        self.data = data
        self.label = label
        self.dev_range = dev_range

        while True:
            A = np.zeros((batchsize, data.shape[1], data.shape[2], data.shape[3]), dtype='float32')
            P = np.zeros((batchsize, data.shape[1], data.shape[2], data.shape[3]), dtype='float32')
            N = np.zeros((self.num_neg, batchsize, data.shape[1], data.shape[2], data.shape[3]), dtype='float32')

            for i in range(batchsize):
                if npair_type == 'samedev':
                    a, p, n = self.get_npair_samedev()
                elif npair_type == 'diffdev':
                    a, p, n = self.get_npair_diffdev()
                else: 
                    raise Exception('Invalid npair type. Accepted values: [samedev, diffdev]') 

                A[i, :, :, :] = a
                P[i, :, :, :] = p
                for neg_i in np.arange(self.num_neg):
                    N[neg_i, i, :, :, :] = n[neg_i]

            yield_label = np.ones(batchsize) # dummy label, not actually used
            yield_data = [A, P]
            for neg_i in np.arange(self.num_neg):
                N_i = np.squeeze(N[neg_i, :, :, :, :])
                yield_data.append(N_i)
            
            # In triple loss, yield_data is [A, P, N]
            # In N-loss, yield data is [A, P, N1, ..., NN]
            yield yield_data, yield_label

    # TODO: consider an idea of adding npair picker with a mix ratio
    
    # This function will pick N negative samples from the same device
    def get_npair_samedev(self):
        # Randomly pick a device for anchor a
        a = self.dev_range[np.random.randint(len(self.dev_range))]
        
        # Find a device other than anchor for negative sample
        n = a # Assign that device for n to start searching for a different device
        while n == a: # keep searching while you find a different device
            n = self.dev_range[np.random.randint(len(self.dev_range))]

        # Retrieve samples for our anchor, positive and negative devices
        anchor = self.call_sample(a)
        positive = self.call_sample(a) # note: positive must have the same device as anchor
        negatives = [self.call_sample(n) for _ in np.arange(self.num_neg)]

        return anchor, positive, negatives

    # This function will pick N negative samples from different devices
    def get_npair_diffdev(self):
        # Randomly pick a device for anchor a
        a = self.dev_range[np.random.randint(len(self.dev_range))]

        # Find N unique devices other than anchor for negative sample
        n = []
        n_dev = self.dev_range[np.random.randint(len(self.dev_range))]
        # Generate num_neg unique devices for our negatives
        for _ in np.arange(self.num_neg):
            while n_dev in n or n_dev == a:
                n_dev = self.dev_range[np.random.randint(len(self.dev_range))]
            n.append(n_dev)

        # Retrieve samples for our anchor, positive and negative devices
        anchor = self.call_sample(a)
        positive = self.call_sample(a) # note: positive must have the same device as anchor
        negatives = [self.call_sample(n_dev) for n_dev in n]

        return anchor, positive, np.array(negatives)

    def call_sample(self, label_name):
        # TODO: consider an "except" feature to make sure randomizer doesn't pick up the same sample in one go
        # Choose an image from our training or test data with the given label.
        num_sample = len(self.label)
        idx = np.random.randint(num_sample)
        while self.label[idx] != label_name:
            # keep searching randomly!
            idx = np.random.randint(num_sample) 
        return self.data[idx]