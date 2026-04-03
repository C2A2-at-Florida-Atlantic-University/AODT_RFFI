import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop
from dataset_preparation import ChannelIndSpectrogram
from deep_learning_models import identity_loss, QuadrupletNet, TripletNet, RNNTripletNet
from keras.models import load_model
import tensorflow as tf
try:
    import seaborn as sea  # noqa: F401
except ImportError:
    sea = None
import matplotlib.pyplot as plt  # noqa: F401

tf.random.set_seed(42)
np.random.seed(42)
try:
    # Avoid XLA/PTX JIT paths that can fail on some host-driver/container stacks.
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

class ExtractorAPI():

    def train(self, data, label, dev_range, model_config, save_path = None):
        batch_size = model_config['batch_size']
        row = model_config['row']
        loss_type = model_config['loss_type']
        backbone = model_config.get('backbone', 'rnn')
        epochs = model_config.get('epochs', 100)
        data = ChannelIndSpectrogram().channel_ind_spectrogram(data, row, enable_ind=model_config['enable_ind'])
        
        if loss_type == 'triplet_loss': 
            alpha = model_config['alpha']

            if backbone == 'rnn':
                netObj = RNNTripletNet(
                    seed=42,
                    gru_units=model_config.get('rnn_gru_units', 256),
                    dropout=model_config.get('rnn_dropout', 0.3),
                    recurrent_dropout=model_config.get('rnn_recurrent_dropout', 0.0),
                    bidirectional=model_config.get('rnn_bidirectional', True),
                    num_layers=model_config.get('rnn_num_layers', 2),
                    embedding_dim=model_config.get('rnn_embedding_dim', 512),
                )
            else:
                netObj = TripletNet()
            feature_extractor = netObj.feature_extractor(data.shape)
            net = netObj.create_net(feature_extractor, alpha=alpha)
        elif loss_type == 'quadruplet_loss': 
            alpha = model_config['alpha']
            beta = model_config['beta'] if 'beta' in model_config else 0

            netObj = QuadrupletNet()
            feature_extractor = netObj.feature_extractor(data.shape)
            net = netObj.create_net(feature_extractor, alpha1=alpha, alpha2=beta)
        else: 
            print('Invalid loss type.')
            return None
        
        # Create callbacks during training
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0, patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6, verbose=1),
        ]
        
        # Split dataset into training and validation, preserving class coverage when possible.
        label_flat = np.asarray(label).reshape(-1)
        unique, counts = np.unique(label_flat, return_counts=True)
        can_stratify = (len(unique) >= 2) and np.all(counts >= 2)
        split_kwargs = dict(test_size=0.2, shuffle=True, random_state=42)
        if can_stratify:
            split_kwargs["stratify"] = label_flat
        data_train, data_valid, label_train, label_valid = train_test_split(data, label, **split_kwargs)

        # Triplet validation requires at least 2 classes in validation split.
        if len(np.unique(np.asarray(label_valid).reshape(-1))) < 2:
            print("[WARN] Validation split has <2 classes; reusing training split for validation.")
            data_valid, label_valid = data_train, label_train

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
        steps_per_epoch = max(1, data_train.shape[0] // batch_size)
        validation_steps = max(1, data_valid.shape[0] // batch_size)
        history = net.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=validation_steps,
            verbose=1,
            callbacks=callbacks,
            shuffle=False,
        )

        if save_path:
            feature_extractor.save(save_path, overwrite=True)

        return feature_extractor, history

    def load(self, model_path, compile=False):
        return load_model(model_path, compile, safe_mode=False)

    def load_feature_extractor(self, model_path, model_config, datashape):
        """Reconstruct the feature extractor architecture and load saved weights."""
        backbone = model_config.get('backbone', 'rnn')
        loss_type = model_config.get('loss_type', 'triplet_loss')

        if loss_type in ('triplet_loss',):
            if backbone == 'rnn':
                netObj = RNNTripletNet(
                    seed=42,
                    gru_units=model_config.get('rnn_gru_units', 256),
                    dropout=model_config.get('rnn_dropout', 0.3),
                    recurrent_dropout=model_config.get('rnn_recurrent_dropout', 0.0),
                    bidirectional=model_config.get('rnn_bidirectional', True),
                    num_layers=model_config.get('rnn_num_layers', 2),
                    embedding_dim=model_config.get('rnn_embedding_dim', 512),
                )
            else:
                netObj = TripletNet()
        elif loss_type == 'quadruplet_loss':
            netObj = QuadrupletNet()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        feature_extractor = netObj.feature_extractor(datashape)
        feature_extractor.load_weights(model_path)
        return feature_extractor

    def run(self, model, data, model_config):
        # Prepare input data for the model (convert to spectrogram images)
        data_freq = ChannelIndSpectrogram().channel_ind_spectrogram(data, model_config['row'], enable_ind=model_config['enable_ind'])

        # Extract fingerprints from the trained model
        return model.predict(data_freq, verbose=0) #, data_freq

# Example usage
if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")