import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop
from dataset_preparation import ChannelIndSpectrogram
from deep_learning_models import NPairNet, identity_loss
from singleton import Singleton
from keras.models import load_model
from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class ExtractorAPI(metaclass=Singleton):

    def train(self, data, label, dev_range, model_config, save_path = None):
        batch_size = model_config['batch_size']
        patience = model_config['patience']
        row = model_config['row']
        loss_type = model_config['loss_type']
        alpha = model_config['alpha']
        num_neg = model_config['loss_num_neg']
        npair_type = model_config['npair_type']
        
        ChannelIndSpectrogramObj = ChannelIndSpectrogram()
        
        # Convert time-domain IQ samples to channel-independent spectrograms.
        data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data, row)
        
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
        data_freq = ChannelIndSpectrogram().channel_ind_spectrogram(data, model_config['row'])

        # Extract fingerprints from the trained model
        return model.predict(data_freq)

    def evaluate_closed_set_knn(self, model, data_epoch_1, labels_epoch_1, data_epoch_2, labels_epoch_2, model_config, render_confusion_matrix=True):
        # Produce fingerprints for the epoch #1
        fps_epoch_1 = self.run(model, data_epoch_1, model_config)

        # Perform the enrollment: fit a KNN classifier based on produced fingerprints
        classifier = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
        classifier.fit(fps_epoch_1, np.ravel(labels_epoch_1))

        # Produce fingerprints for the epoch #2
        fps_epoch_2 = self.run(model, data_epoch_2, model_config)
        labels_epoch_2_predicted = classifier.predict(fps_epoch_2)

        # Get the accuracy
        accuracy = accuracy_score(labels_epoch_2, labels_epoch_2_predicted)
        
        if render_confusion_matrix:
            conf_matrix = confusion_matrix(labels_epoch_2, labels_epoch_2_predicted)
            plt.figure(figsize=(12, 10), dpi=60)
            # TODO: sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu', xticklabels=device_ids, yticklabels=device_ids)
            sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu')
            
            plt.title(f'Device Confusion Matrix (Euclidean Distance)')
            plt.xlabel('Device ID')
            plt.ylabel('Device ID')
            plt.tight_layout()
            plt.show()

        return accuracy

    def evaluate_closed_set_custom_multirx(self, models, rx_ids, data_epoch_1_multirx, labels_epoch_1, data_epoch_2_multirx, labels_epoch_2, rssi_epoch_2_multirx, model_config, render_confusion_matrix=True):
        # Produce fingerprints for the epoch #1
        fps_epoch_1 = self.run(model, data_epoch_1, model_config)

        # Produce fingerprints for the epoch #2
        fps_epoch_2 = self.run(model, data_epoch_2, model_config)

        

# Example usage
if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")