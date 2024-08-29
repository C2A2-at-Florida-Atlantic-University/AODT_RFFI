import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import utility
import threading
from dotenv import load_dotenv

load_dotenv()

class MilvusClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(MilvusClient, cls).__new__(cls)
        return cls._instance

    def initialize(self, collection):
        # Connect to Zilliz cloud
        connections.connect(
            alias="default", 
            uri=os.getenv("ZILLIZ_URI"), 
            token=os.getenv("ZILLIZ_API_KEY"))
        
        self.collection_name = collection
        
        # # Define schema
        # fields = [
        #     FieldSchema(name="primary_key", dtype=DataType.INT64, is_primary=True),
        #     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),
        #     FieldSchema(name="device_id", dtype=DataType.INT16),
        #     FieldSchema(name="device_name", dtype=DataType.VARCHAR, max_length=100),
        #     FieldSchema(name="epoch_no", dtype=DataType.INT16)
        # ]
        # schema = CollectionSchema(fields, "Device collection schema")
        
        # # Create collection if not exists
        # if not utility.has_collection(self.collection_name):
        #     self.collection = Collection(name=self.collection_name, schema=schema)
        # else:
        self.collection = Collection(name=self.collection_name)

    def enroll(self, primary_key, vector, device_id, device_name, epoch_no):
        data = [
            [primary_key],         # primary_key
            [vector],              # vector
            [device_id],           # device_id
            [device_name],         # device_name
            [epoch_no]             # epoch_no
        ]
        self.collection.insert(data)

    def find(self, vector):
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[vector], 
            anns_field="vector", 
            param=search_params, 
            limit=1, 
            expr=None, 
            output_fields=["primary_key", "device_id", "device_name", "epoch_no"]
        )
        
        if results[0]:
            return {
                "primary_key": results[0][0].id,
                "device_id": results[0][0].entity.get("device_id"),
                "device_name": results[0][0].entity.get("device_name"),
                "epoch_no": results[0][0].entity.get("epoch_no")
            }
        else:
            return "Device not found"

if __name__ == '__main__':
    # Usage
    milvus_client = MilvusClient()
    milvus_client.initialize("orbit_dataset_v2_jul19")

    # # Enroll a new device
    # milvus_client.enroll(
    #     vector=[0.1] * 512, 
    #     device_id=123, 
    #     device_name="Device A", 
    #     epoch_no=1
    # )

    # # Find a device by vector
    result = milvus_client.find([0.1] * 512)
    print(result)