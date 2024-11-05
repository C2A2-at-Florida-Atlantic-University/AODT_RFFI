import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam
from dataset_preparation import ChannelIndSpectrogram
from deep_learning_models import identity_loss, QuadrupletNet, TripletNet
from keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sea
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

class ExtractorAPI():

    def train(self, data, label, dev_range, model_config, save_path = None):
        batch_size = model_config['batch_size']
        row = model_config['row']
        loss_type = model_config['loss_type']
        alpha = model_config['alpha']
        
        data = ChannelIndSpectrogram().channel_ind_spectrogram(data, row, enable_ind=model_config['enable_ind'])
        
        if loss_type == 'triplet_loss': 
            netObj = TripletNet()
            feature_extractor = netObj.feature_extractor(data.shape)
            net = netObj.create_net(feature_extractor, alpha=alpha)
        elif loss_type == 'quadruplet_loss': 
            netObj = QuadrupletNet()
            feature_extractor = netObj.feature_extractor(data.shape)
            net = netObj.create_net(feature_extractor, alpha1=alpha, alpha2=alpha/3)
        else: 
            print('Invalid loss type.')
            return None
        
        # Create callbacks during training
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True), 
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)]
        
        # Split the dasetset into validation and training sets.
        data_train, data_valid, label_train, label_valid = train_test_split(data, label, test_size=0.2, shuffle=True, random_state=42)

        del data, label
        
        # Create the trainining generator.
        train_generator = netObj.create_generator(batch_size, dev_range,  data_train, label_train)
        # Create the validation generator.
        valid_generator = netObj.create_generator(batch_size, dev_range, data_valid, label_valid)
        
        # Use the RMSprop optimizer for training.
        opt = RMSprop(learning_rate=1e-3)
        # opt = Adam(learning_rate=1e-3)
        net.compile(loss = identity_loss, optimizer = opt)

        # Start training.
        history = net.fit(train_generator,
                                steps_per_epoch = data_train.shape[0]//batch_size,
                                epochs = 1000,
                                validation_data = valid_generator,
                                validation_steps = data_valid.shape[0]//batch_size,
                                verbose=1, 
                                callbacks = callbacks,
                                workers=1,
                                use_multiprocessing=False,
                                shuffle=False)

        if save_path:
            feature_extractor.save(save_path, overwrite=True)

        return feature_extractor, history

    def load(self, model_path, compile=False):
        return load_model(model_path, compile, safe_mode=False)

    def run(self, model, data, model_config):
        # Prepare input data for the model (convert to spectrogram images)
        data_freq = ChannelIndSpectrogram().channel_ind_spectrogram(data, model_config['row'], enable_ind=model_config['enable_ind'])

        # Extract fingerprints from the trained model
        return model.predict(data_freq, verbose=0), data_freq

# Example usage
if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")