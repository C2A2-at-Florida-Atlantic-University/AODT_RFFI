from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop
from dataset_preparation import ChannelIndSpectrogram
from deep_learning_models import NPairNet, identity_loss
from Singleton import Singleton
from keras.models import load_model

class ExtractorAPI(metaclass=Singleton):

    def train(self, data, label, dev_range, model_config, save_path = None):
        batch_size = model_config['batch_size']
        patience = model_config['patience']
        row = model_config['row']
        col = model_config['col']
        loss_type = model_config['loss_type']
        alpha = model_config['alpha']
        num_neg = model_config['loss_num_neg']
        npair_type = model_config['npair_type']
        
        ChannelIndSpectrogramObj = ChannelIndSpectrogram()
        
        # Convert time-domain IQ samples to channel-independent spectrograms.
        data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data, row, col)
        
        NPairNetObj = NPairNet()
        
        # Create an RFF extractor.
        feature_extractor = NPairNetObj.feature_extractor(data.shape)
        
        # Create the Triplet net using the RFF extractor.
        npair_net = NPairNetObj.create_npair_net(feature_extractor, alpha, num_neg, loss_type)

        # Create callbacks during training
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta = 0, patience = patience, restore_best_weights=True), 
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = patience, min_lr=1e-6, verbose=1)]
        
        # Split the dasetset into validation and training sets.
        data_train, data_valid, label_train, label_valid = train_test_split(data, label, test_size=0.1, shuffle= True)
        del data, label
        
        # Create the trainining generator.
        train_generator = NPairNetObj.create_generator(batch_size, dev_range,  data_train, label_train, npair_type)
        # Create the validation generator.
        valid_generator = NPairNetObj.create_generator(batch_size, dev_range, data_valid, label_valid, npair_type)
        
        # Use the RMSprop optimizer for training.
        opt = RMSprop(learning_rate=1e-3)
        npair_net.compile(loss = identity_loss, optimizer = opt)

        # Start training.
        history = npair_net.fit(train_generator,
                                steps_per_epoch = data_train.shape[0]//batch_size,
                                epochs = 1000,
                                validation_data = valid_generator,
                                validation_steps = data_valid.shape[0]//batch_size,
                                verbose=1, 
                                callbacks = callbacks)

        if save_path:
            feature_extractor.save(save_path, overwrite=True)

        return feature_extractor, history

    def load(self, model_path, compile=False):
        return load_model(model_path, compile, safe_mode=False)

    def run(self, model, data, model_config):
        # Prepare input data for the model (convert to spectrogram images)
        data_freq = ChannelIndSpectrogram().channel_ind_spectrogram(data, model_config['row'], model_config['col'])

        # Extract fingerprints from the trained model
        return model.predict(data_freq)

# Example usage
if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")