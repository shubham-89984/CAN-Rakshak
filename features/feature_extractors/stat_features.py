from features.feature_extractors.base import FeatureExtractor
from utilities import *
import os
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class Stat(FeatureExtractor):
    def __init__(self, cfg=None):
        super().__init__(cfg or {})
        self.X, self.Y = self.extract_features(cfg=cfg or {})
        print("Features extracted using Stat feature extractor")
        print("X shape : ", self.X.shape)
        print("Y shape : ", self.Y.shape)

    def extract_features(self, cfg=None):
        cfg        = cfg or {}
        model_name = cfg.get('model_name', '')
        mode       = cfg.get('mode', 'train')

        print("Extracting features")
        df = self.read_attack_data(self.file_path)

        X = df.drop(columns=['flag', 'timestamp']).values
        Y = (df['flag'].values == 'T').astype(int)

        # Save raw features so the splitter can split them
        self._save_raw_features(X, Y)

        scalar_path = os.path.join(self.dataset_path, model_name + "scalar.pkl")

        if mode == "train":
            scaler = StandardScaler()
            scaler.fit(X)
            joblib.dump(scaler, scalar_path)

        if mode == "test":
            scaler = joblib.load(scalar_path)

        X = scaler.transform(X)

        return X, Y

    def _save_raw_features(self, X, Y):
        stat_dir = os.path.join(self.features_path, "Stat")
        os.makedirs(stat_dir, exist_ok=True)

        prefix       = os.path.splitext(self.file_name)[0]
        features_csv = os.path.join(stat_dir, prefix + "_features.csv")
        labels_csv   = os.path.join(stat_dir, prefix + "_labels.csv")

        np.savetxt(features_csv, X, delimiter=",")

        with open(labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "label"])
            for i, label in enumerate(Y):
                writer.writerow([i, label])

        print(f"  Saved          : {os.path.basename(features_csv)}")
        print(f"  Saved          : {os.path.basename(labels_csv)}")
        
    def read_attack_data(self,data_path):

        columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4',
            'data5', 'data6', 'data7', 'flag']

        data = pd.read_csv(data_path, names = columns,skiprows=1)
        data = shift_columns(data)

        data = data.replace(np.nan, '00')

        data_cols = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']

        data[data_cols] = data[data_cols].astype(str)
        data['data'] = data[data_cols].apply(''.join, axis=1)
        data.drop(columns = data_cols, inplace = True, axis = 1)

        data['can_id'] = data['can_id'].apply(hex_to_dec)
        data['data'] = data['data'].apply(hex_to_dec)

        data = data.assign(IAT=data['timestamp'].diff().fillna(0))

        return data
