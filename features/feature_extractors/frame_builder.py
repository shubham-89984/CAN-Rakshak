import os
from features.feature_extractors.base import FeatureExtractor
import csv
import numpy as np


class FrameBuilder(FeatureExtractor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.extract_features(self.file_path)

    def extract_features(self, file_path):
        self.build_frames(file_path, self.features_path, 29, 29)

    def build_frames(self, csv_file, features_path, rows, bits):
        packets = []
        labels  = []
        traffic_rows = []

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                traffic_rows.append(row)

                try:
                    can_hex   = row[1][:4]
                    bitstring = format(int(can_hex, 16), f"0{bits}b")
                    assert int(bitstring, 2) == int(can_hex, 16), \
                        f"Mismatch: {bitstring} vs {can_hex}"
                except:
                    continue

                packets.append(bitstring)

                lbl = row[-1].strip().upper()
                labels.append(1 if lbl in ["T", "1", "ATTACK", "A"] else 0)

        total     = len(packets)
        num_frames = total // rows

        frames, frame_labels = [], []

        for i in range(num_frames):
            block = packets[i * rows:(i + 1) * rows]
            frame = np.array([[int(b) for b in s] for s in block]).reshape(rows, bits, 1)
            frames.append(frame)
            frame_labels.append(max(labels[i * rows:(i + 1) * rows]))

        print(f"  Input          : {os.path.basename(csv_file)}")
        print(f"  Frames         : {num_frames} (Benign: {frame_labels.count(0)}, Attack: {frame_labels.count(1)})")

        features_dir = os.path.join(features_path, "Frames")
        os.makedirs(features_dir, exist_ok=True)
        frames_csv = os.path.join(features_dir, self.file_name[:-4] + "_frames.csv")
        labels_csv = os.path.join(features_dir, self.file_name[:-4] + "_labels.csv")
        self.save_frames_and_labels(frames, frame_labels, frames_csv, labels_csv)

    def save_frames_to_csv(self, frames, frame_labels, out_csv):
        num_frames, rows, bits, _ = frames.shape

        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(num_frames):
                writer.writerow([f"Frame {i}", f"Label={frame_labels[i]}"])
                for r in range(rows):
                    writer.writerow(frames[i][r, :, 0].tolist())
                writer.writerow([])

        print(f"Saved frames to {out_csv}")

    def save_frames_and_labels(self, frames, frame_labels, frames_csv, labels_csv):
        frames       = np.array(frames)
        frame_labels = np.array(frame_labels)
        num_frames, rows, bits, _ = frames.shape

        with open(frames_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(num_frames):
                for r in range(rows):
                    writer.writerow(frames[i][r, :, 0].tolist())

        print(f"  Saved          : {os.path.basename(frames_csv)}")

        with open(labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "label"])
            for i, label in enumerate(frame_labels):
                writer.writerow([i, label])

        print(f"  Saved          : {os.path.basename(labels_csv)}")
