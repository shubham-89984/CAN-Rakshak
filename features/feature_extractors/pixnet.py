import os
from features.feature_extractors.base import FeatureExtractor
from features.image.data_frame import convert_to_json
from features.image.traffic_encoder import generate_image


class PixNet(FeatureExtractor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.extract_features(self.file_path)

    def extract_features(self, file_path):
        images_output_dir = os.path.join(self.features_path, "Images", self.file_name[:-4] + "_images")
        csv_dir = os.path.join(self.dir_path, "..", "datasets", self.dataset_name, "csv_files")
        os.makedirs(csv_dir, exist_ok=True)
        csv_track = os.path.join(csv_dir, self.file_name[:-4] + "_track.csv")

        print(f"  Input          : {os.path.basename(file_path)}")
        print(f"  Images dir     : {images_output_dir}")
        convert_to_json(file_path, self.json_file_path)
        generate_image(self.json_file_path, images_output_dir, csv_track)
