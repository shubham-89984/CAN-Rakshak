import abc
import os


class FeatureExtractor:
    def __init__(self, cfg):
        self.dir_path               = cfg['dir_path']
        self.dataset_name           = cfg['dataset_name']
        self.file_name              = cfg['file_name']
        self.feature_extractor_name = cfg['feature_extractor']
        self.feature_extraction     = cfg['feature_extraction']

        self.dataset_path = os.path.join(self.dir_path, "..", "datasets", self.dataset_name, "modified_dataset")
        os.makedirs(self.dataset_path, exist_ok=True)

        self.csv_file_name = next(
            (self.file_name.replace(ext, ".csv") for ext in [".log", ".txt", ".csv"] if self.file_name.endswith(ext)),
            self.file_name
        )
        self.file_path = os.path.join(self.dataset_path, self.csv_file_name)

        self.json_folder = os.path.join(self.dir_path, "..", "datasets", self.dataset_name, "json_files")
        os.makedirs(self.json_folder, exist_ok=True)

        self.json_file_name = self.csv_file_name[:-4] + ".json"
        self.json_file_path = os.path.join(self.json_folder, self.json_file_name)
        self.features_path  = os.path.join(self.dataset_path, "..", "features")
        os.makedirs(self.features_path, exist_ok=True)

    @abc.abstractmethod
    def extract(self):
        """Base method — override in subclasses"""
        raise NotImplementedError("Subclasses must implement this method")
