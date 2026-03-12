import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import utils
from dataset_api import DatasetAPI
from extractor_api import ExtractorAPI
from dataset_preparation import awgn
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class EvaluationAPI():

    def __init__(self, rx_ids, data_config, aug_config, model_config, root_dir, matlab_src_dir, matlab_session_id, aug_on):
        self.rx_ids = rx_ids
        self.aug_on = aug_on
        self.data_config = data_config
        self.aug_config = aug_config
        self.model_config = model_config
        self.dataset_api = DatasetAPI(root_dir, matlab_src_dir, matlab_session_id, aug_on)
        self.extractor_api = ExtractorAPI()

    def evaluate_aodt_hf_closed_set(self, rx_id=None, k=10, fig_path=None, apply_noise=False, use_pretrained=True):
        """
        Closed-set evaluation for AODT Hugging Face datasets.
        Requires data_config:
          - dataset_name='aodt_hf'
          - hf_repo_id='namespace/dataset'
        Optional:
          - hf_train_split / hf_test_split OR hf_train_ratio
          - hf_* filters (batches/slots), hf_sym_mode, hf_rx_ant
          - model_path (directory for extractor_*.keras)
        """
        if self.data_config['dataset_name'] != DatasetAPI.DATASET_AODT_HF:
            raise ValueError("evaluate_aodt_hf_closed_set requires dataset_name='aodt_hf'")

        target_rx = rx_id if rx_id is not None else self.rx_ids[0]
        model_dir = self.data_config.get('model_path', os.path.join(self.dataset_api.root_dir, 'aodt_hf_models'))
        model_file = os.path.join(model_dir, f"extractor_{target_rx}.keras")

        (
            data_train,
            labels_train,
            _,
            data_test,
            labels_test,
            _,
            node_ids_train,
            node_ids_test,
        ) = self.dataset_api.load_hf_train_test(self.data_config, shuffle_train=True, shuffle_test=False)

        data_train = data_train[:, 0:self.data_config['samples_count']]
        data_test = data_test[:, 0:self.data_config['samples_count']]

        # Keep only labels present in both splits for a valid closed-set setup.
        shared_labels = sorted(list(set(node_ids_train).intersection(set(node_ids_test))))
        if not shared_labels:
            raise RuntimeError("No overlapping labels between train/test HF splits for closed-set evaluation.")

        train_mask = np.isin(labels_train.flatten(), shared_labels)
        test_mask = np.isin(labels_test.flatten(), shared_labels)
        data_train = data_train[train_mask]
        labels_train = labels_train[train_mask]
        data_test = data_test[test_mask]
        labels_test = labels_test[test_mask]

        if apply_noise:
            data_train = awgn(data_train, np.arange(self.aug_config['awgn'][0][0], self.aug_config['awgn'][0][1]))
            data_test = awgn(data_test, np.arange(self.aug_config['awgn'][0][0], self.aug_config['awgn'][0][1]))

        if use_pretrained and os.path.exists(model_file):
            feature_extractor = self.extractor_api.load(model_file, compile=False)
        else:
            os.makedirs(model_dir, exist_ok=True)
            feature_extractor, _ = self.extractor_api.train(
                data_train,
                labels_train,
                shared_labels,
                self.model_config,
                save_path=model_file,
            )

        return self.evaluate_closed_set_knn(
            feature_extractor,
            data_train,
            labels_train,
            data_test,
            labels_test,
            k=k,
            fig_path=fig_path,
        )

    def evaluate_closed_set_knn(self, model, data_epoch_1, labels_epoch_1, data_epoch_2, labels_epoch_2, k=10, fig_path=None):
        epoch_1_device_ids = set(labels_epoch_1.flatten())
        epoch_2_device_ids = set(labels_epoch_2.flatten())

        if epoch_1_device_ids == epoch_2_device_ids:
            print("Great! Epoch #1 and epoch #2 contain identical sets of device IDs. We can perform closed-set evaluation.")
        else:
            print("The device IDs in Epoch #2 and Epoch #1 must be identical. Cannot proceed.")
            return -1

        # Produce fingerprints for the epoch #1
        fps_epoch_1 = self.extractor_api.run(model, data_epoch_1, self.model_config)

        # Perform the enrollment: fit a KNN classifier based on produced fingerprints
        classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        classifier.fit(fps_epoch_1, np.ravel(labels_epoch_1))

        # Produce fingerprints for the epoch #2
        fps_epoch_2 = self.extractor_api.run(model, data_epoch_2, self.model_config)
        labels_epoch_2_predicted = classifier.predict(fps_epoch_2)
        
        # Get the accuracy
        accuracy = accuracy_score(labels_epoch_2, labels_epoch_2_predicted)
        
        if fig_path:
            device_ids = sorted(list(set(labels_epoch_2.flatten())))

            conf_matrix = confusion_matrix(labels_epoch_2, labels_epoch_2_predicted, labels=device_ids)
            conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
            annotations = np.array([["{:.0f}%".format(value) for value in row] for row in conf_matrix_percent])

            utils.apply_ieee_style()
            plt.figure(figsize=(10, 8), dpi=80)
            sns.heatmap(conf_matrix_percent, annot=annotations, fmt="", cmap='YlGnBu', xticklabels=device_ids, yticklabels=device_ids)
            plt.xlabel('Predicted Device ID')
            plt.ylabel('True Device ID')
            plt.gca().xaxis.tick_top()
            plt.gca().xaxis.set_label_position('top') 

            if fig_path: plt.savefig(fig_path, format='eps', bbox_inches='tight', pad_inches=0.1)

            plt.title(f'Closed-set classification accuracy: {np.round(accuracy * 100, 2)}%')
            plt.tight_layout()
            plt.show()

        return accuracy, labels_epoch_2, labels_epoch_2_predicted
    
    def evaluate_open_set_knn(self, model, data_epoch_1, labels_epoch_1, data_epoch_2, labels_epoch_2, model_config, fig_path=None):
        # Here, we also expect two epochs. But we expect that the number set of devices in epoch #1 will be smaller compared to
        # the set of devices in epoch #2.
        epoch_1_device_ids = set(labels_epoch_1.flatten())
        epoch_2_device_ids = set(labels_epoch_2.flatten())

        if epoch_1_device_ids <= epoch_2_device_ids:
            print("Great! Epoch #2 contains more devices than #1, and #1 is a subset of #2. We can start open-set evaluation.")
        else:
            print("Device IDs in epoch #1 must be a subset of device IDs in epoch #2. Cannot proceed.")
            return -1

        # Produce fingerprints for the epoch #1
        fps_epoch_1 = self.extractor_api.run(model, data_epoch_1, model_config)

        # Perform the enrollment: fit a KNN classifier based on produced fingerprints
        classifier = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
        classifier.fit(fps_epoch_1, np.ravel(labels_epoch_1))

        # Produce fingerprints for the epoch #2
        fps_epoch_2 = self.extractor_api.run(model, data_epoch_2, model_config)

        # Find the nearest 15 neighbors in the RFF database and calculate the distances to them.
        distances, _ = classifier.kneighbors(fps_epoch_2)
        
        # Calculate the average distance to the nearest 15 neighbors.
        detection_score = distances.mean(axis=1)
  
        # Create a mask array which will contain 1 if device is from enrolled list, and 0 if it's new
        true_labels = [1 if item in epoch_1_device_ids else 0 for item in labels_epoch_2.flatten()]

        # Compute receiver operating characteristic (ROC).
        fpr, tpr, _ = roc_curve(true_labels, detection_score, pos_label = 1)

        # Invert false positive and true positive ratios to convert from distances to probabilities
        fpr = 1 - fpr  
        tpr = 1 - tpr

        # Compute EER
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        # Compute AUC
        roc_auc = auc(fpr, tpr)

        if fig_path:
            eer_point = min(zip(fpr, tpr), key=lambda x: abs(x[0] - (1-x[1])))

            utils.apply_ieee_style()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
            plt.plot(eer_point[0], eer_point[1], 'ro', markersize=10, label=f'EER = {eer:.2f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            if fig_path: plt.savefig(fig_path, format='eps', bbox_inches='tight', pad_inches=0.1)
            plt.show()

if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")