import os
import uuid
import numpy as np
import chromadb
from chromadb import Settings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.spatial import distance
from dataset_preparation import awgn
from dataset_api import DatasetAPI
from extractor_api import ExtractorAPI

np.random.seed(42)

class FingerprintingAPI():

    VECTOR_SEARCH_MAX_RESULTS = 3

    def __init__(self, rx_ids, data_config, aug_config, model_config, root_dir, matlab_src_dir, matlab_session_id, aug_on=False):
        self.rx_ids = rx_ids
        self.aug_on = aug_on

        self.data_config = data_config
        self.aug_config = aug_config
        self.model_config = model_config

        self.dataset_api = DatasetAPI(root_dir, matlab_src_dir, matlab_session_id, aug_on)
        self.extractor_api = ExtractorAPI()
        self.db_client = chromadb.Client(Settings(allow_reset=True))

        self.models = {rx_id : None for rx_id in rx_ids}
        self.db_collections = {rx_id: f"collection_{rx_id}" for rx_id in rx_ids}

    def purge_database(self):
        for collection in self.db_client.list_collections():
            self.db_client.delete_collection(name=collection.name)

    def list_enrolled_devices(self, render_confusion_matrices=True):
        all_devices = set()
        device_info = {}

        for rx_id, collection_name in self.db_collections.items():
            collection = self.db_client.get_or_create_collection(collection_name)
            
            # Get all entries from the collection, including embeddings
            results = collection.get(ids=[], include=["metadatas", "embeddings"])

            for i, device_id in enumerate(results['ids']):
                all_devices.add(device_id)
                metadata = results['metadatas'][i]
                embedding = results['embeddings'][i]
                
                if device_id not in device_info:
                    device_info[device_id] = {
                        'date_added': metadata['date_added'],
                        'date_updated': metadata['date_updated'] if metadata['date_updated'] else 'Not updated',
                        'embeddings': {rx_id: embedding}
                    }
                else:
                    # Update date_updated if necessary
                    if metadata['date_updated'] and (device_info[device_id]['date_updated'] == 'Not updated' or 
                    metadata['date_updated'] > device_info[device_id]['date_updated']):
                        device_info[device_id]['date_updated'] = metadata['date_updated']
                    
                    # Add embedding for this receiver
                    device_info[device_id]['embeddings'][rx_id] = embedding

        print(f"Total number of unique devices: {len(all_devices)}\n")
        print("Device Statistics:")
        if len(device_info) == 0:
            print("No devices found.")
        else:
            for device_id, info in device_info.items():
                print(f"Device ID: {device_id} (added: {info['date_added']}, last updated: {info['date_updated']})")

        if render_confusion_matrices:
            for rx_id in self.rx_ids:
                self._render_confusion_matrix(rx_id, device_info)

        return device_info

    def _render_confusion_matrix(self, rx_id, device_info):
        # 1. Extract embeddings & device IDs for the confusion matrix
        device_ids = list(device_info.keys())
        embeddings = [device_info[device_id]['embeddings'][rx_id] for device_id in device_ids]

        # 2. Produce a confusion matrix with respect to a given receiver
        confusion_matrix = np.zeros((len(device_ids), len(device_ids)))
        
        # Calculate Euclidean distances and fill the confusion matrix
        for i in np.arange(len(device_ids)):
            for j in np.arange(len(device_ids)):
                confusion_matrix[i, j] = distance.euclidean(embeddings[i], embeddings[j])
        
        # Create a heatmap
        plt.figure(figsize=(12, 10), dpi=60)
        sns.heatmap(confusion_matrix, annot=True, cmap='YlGnBu', xticklabels=device_ids, yticklabels=device_ids)
        
        plt.title(f'Device Confusion Matrix (Euclidean Distance) for Receiver {rx_id}')
        plt.xlabel('Device ID')
        plt.ylabel('Device ID')
        plt.tight_layout()
        plt.show()

    def train_models(self, apply_noise=False):
        if self.data_config['dataset_name'] != DatasetAPI.DATASET_AODT_HF:
            raise ValueError(
                "This cleaned pipeline is HF-only. Set data_config['dataset_name'] to DatasetAPI.DATASET_AODT_HF."
            )

        train_histories = {}
        data, label, _, _, _, _, node_ids_train, _ = self.dataset_api.load_hf_train_test(
            self.data_config, shuffle_train=True, shuffle_test=False
        )
        data = data[:, 0:self.data_config['samples_count']]

        if apply_noise:
            data = awgn(data, np.arange(self.aug_config['awgn'][0][0], self.aug_config['awgn'][0][1]))

        model_dir = self.data_config.get('model_path', os.path.join(self.dataset_api.root_dir, 'aodt_hf_models'))
        os.makedirs(model_dir, exist_ok=True)

        for rx_id in self.rx_ids:
            model_save_path = os.path.join(model_dir, f"extractor_{rx_id}.keras")
            feature_extractor, history = self.extractor_api.train(
                data, label, node_ids_train, self.model_config, save_path=model_save_path
            )
            train_histories[rx_id] = history
            self.models[rx_id] = feature_extractor

        return self.models, train_histories

    def load_models(self):
        if self.data_config['dataset_name'] != DatasetAPI.DATASET_AODT_HF:
            raise ValueError(
                "This cleaned pipeline is HF-only. Set data_config['dataset_name'] to DatasetAPI.DATASET_AODT_HF."
            )

        model_dir = self.data_config.get('model_path', os.path.join(self.dataset_api.root_dir, 'aodt_hf_models'))
        for rx_id in self.rx_ids:
            self.models[rx_id] = self.extractor_api.load(
                os.path.join(model_dir, f"extractor_{rx_id}.keras"), compile=False
            )
        return self.models

    def _db_insert_device(self, rx_fps, rx_rssis):
        new_device_id = str(uuid.uuid4())
        insert_date = datetime.now().isoformat()

        for rx_id in self.rx_ids:
            rx_db_collection = self.db_client.get_or_create_collection(self.db_collections[rx_id])

            rx_db_collection.add(
                embeddings=[rx_fps[rx_id].tolist()],
                ids=[new_device_id],
                metadatas=[{
                    "rssi": rx_rssis[rx_id] if rx_rssis else "N/A",
                    "date_added": insert_date,
                    "date_updated": ""
                }])

        return new_device_id

    def _db_update_device(self, known_device_id, rx_fps, rx_rssis):
        update_date = datetime.now().isoformat()

        for rx_id in self.rx_ids:
            rx_db_collection = self.db_client.get_or_create_collection(self.db_collections[rx_id])

            rx_db_collection.update(
                embeddings=[rx_fps[rx_id].tolist()],
                ids=[known_device_id],
                metadatas=[{
                    "rssi": rx_rssis[rx_id] if rx_rssis else "N/A",
                    "date_updated": update_date
                }])

        return known_device_id

    def new_signal(self, frames_rx_all, new_device_threshold, apply_noise=False, update_if_known=True, return_distances=False, verbose=True):
        # 1. Initialize a list of device candidates
        device_candidates = {}
        rx_rssis = {}
        rx_fps = {}

        # 2. For each receiver
        for rx_id in self.rx_ids:
            feature_extractor = self.models[rx_id]
            rx_db_collection = self.db_client.get_or_create_collection(self.db_collections[rx_id])

            # Retrieve frames from a given receiver
            frames = frames_rx_all[rx_id]

            fps = np.zeros((len(frames), 512))
            rssis = np.zeros((len(frames), 1))

            # 1. For each of the frames
            for i, frame in enumerate(frames):
                # 1. Pick a specified # of samples
                iq = frame['iq'][0:self.data_config['samples_count']]

                # 2. Optionally, add awgn
                if apply_noise:
                    iq = awgn(np.array([iq]), np.arange(self.aug_config['awgn'][0][0], self.aug_config['awgn'][0][1]))

                # 3. Save frame RSSI value (without transforming for now, to keep its absolute value)
                if 'rssi' in frame:
                    rssis[i] = frame['rssi']
                else: rssis = None

                # 4. Extract a fingerprint
                fps[i, :] = self.extractor_api.run(feature_extractor, np.array([iq]), self.model_config)

            # 2. Aggregate all frame RPs and RSSIs (either by picking one of them, or getting a mean value)
            fp = np.mean(fps, 0)
            rssi = np.mean(rssis) if rssis is not None else None

            # 2.1. Add the RSSI to the dictionary to weigh impact of this receiver
            if rssi: 
                rx_rssis[rx_id] = rssi
            else: rx_rssis = None

            rx_fps[rx_id] = fp

            # 3. Use this fingerprint to look up candidates in the database
            search_results = rx_db_collection.query(
                query_embeddings = [fp.tolist()],
                n_results = FingerprintingAPI.VECTOR_SEARCH_MAX_RESULTS,
                include = ["documents", "metadatas", "distances", "embeddings"])
            search_results_len = len(search_results['ids'][0])
            
            # 4. Did we find any devices?
            for i in np.arange(search_results_len):
                item_id = search_results['ids'][0][i]
                item_document = search_results['documents'][0][i]
                item_metadata = search_results['metadatas'][0][i]
                item_distance = search_results['distances'][0][i]
                if item_id not in device_candidates:
                    device_candidates[item_id] = {
                        'rx_id': rx_id,
                        'document': item_document,
                        'metadata': item_metadata,
                        'distance': item_distance
                    }

        # 3. For each candidate
        candidate_distances = {}
        for candidate_id in device_candidates.keys():
            rx_distances = {}

            # 1. Compute FP distance for each receiver
            for rx_id in self.rx_ids:
                rx_db_collection = self.db_client.get_or_create_collection(self.db_collections[rx_id])
                searh_result = rx_db_collection.get(ids=[candidate_id], include=["embeddings", "metadatas", "documents"])

                candidate_rx_fp = np.array(searh_result['embeddings'][0])
                candidate_rx_distance = distance.euclidean(rx_fps[rx_id], candidate_rx_fp)
                # Some alternatives: distance.cosine, distance.chebyshev, distance.minkowski

                rx_distances[rx_id] = candidate_rx_distance

            # 2. Calculate device candidate weighted distance to our frame
            if rx_rssis:
                candidate_distances[candidate_id] = sum([rx_distances[rx_id] * self.dataset_api.rssi_to_weight(rx_rssis[rx_id]) for rx_id in self.rx_ids])/len(self.rx_ids)
            else: 
                candidate_distances[candidate_id] = np.mean([rx_distances[rx_id] for rx_id in self.rx_ids])

        # 3.1. Return distance to the closest candidate
        if return_distances:
            return min(candidate_distances.values())

        # 3.2. Are we dealing with a known device? (one of distances below threshold)
        response = {}
        if len(candidate_distances) > 0 and min(candidate_distances.values()) < new_device_threshold:
            # Which device is the closest?
            known_device_id = min(candidate_distances, key=candidate_distances.get)
            if update_if_known:
                self._db_update_device(known_device_id, rx_fps, rx_rssis)
            if verbose: print(f"This is a known device. ID: {known_device_id}")
            response['device_hash'] = known_device_id
            response['is_new'] = False
            response['closest_dist'] = min(candidate_distances.values())
        else: # No, this is an unknown device. Add it to all collections
            new_device_id = self._db_insert_device(rx_fps, rx_rssis)
            if verbose: print(f"This is a new device. New ID: {new_device_id}")
            response['device_hash'] = new_device_id
            response['is_new'] = True
            if len(candidate_distances.values()) > 0:
                response['closest_dist'] = min(candidate_distances.values())
            else: response['closest_dist'] = 1

        return response

# Example usage
if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")