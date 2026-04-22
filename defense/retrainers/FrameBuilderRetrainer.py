from common_imports import os, np, pd, tf
from defense.base import BaseDefense
class FrameBuilderRetrainer(BaseDefense):
    """
    Retrainer for datasets processed with the FrameBuilder feature extractor.
    Expects CSV-based frame/label files in the train dataset directory.
    """

    def make_dataset(self, model_path, adversarial_images_path, cfg, adversarial_samples_limit=5000):
        dir_path         = cfg.get('dir_path', '')
        dataset_name     = cfg.get('dataset_name', '')
        file_name        = cfg.get('file_name', '')
        train_dataset_dir_name = cfg.get('train_dataset_dir', '')
        train_dataset_dir = os.path.join(dir_path, "..", "datasets", dataset_name, "train", train_dataset_dir_name)
        train_frames_path = os.path.join(train_dataset_dir, file_name[:-4] + "_train_frames.csv")
        train_labels_path = os.path.join(train_dataset_dir, file_name[:-4] + "_train_labels.csv")

        train_frames = pd.read_csv(train_frames_path, header=None)
        train_labels = pd.read_csv(train_labels_path)

        data       = np.load(adversarial_images_path)
        adv_frames = data["final_test"]
        labels     = data["y_test"]

        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        pred = model.predict(adv_frames)
        pred = np.argmax(pred, axis=1)

        successful_idx = np.where((labels == 1) & (pred == 0))[0]
        if len(successful_idx) > adversarial_samples_limit:
            successful_idx = np.random.choice(successful_idx, adversarial_samples_limit, replace=False)

        adv_success   = adv_frames[successful_idx].squeeze()
        adv_rows      = adv_success.reshape(-1, 29)
        adv_frames_df = pd.DataFrame(adv_rows)

        start_id = train_labels["frame_id"].max() + 1
        adv_labels_df = pd.DataFrame({
            "frame_id": range(start_id, start_id + len(adv_success)),
            "label":    [1] * len(adv_success)
        })

        merged_frames = pd.concat([train_frames, adv_frames_df], ignore_index=True)
        merged_labels = pd.concat([train_labels, adv_labels_df], ignore_index=True)
        merged_labels["frame_id"] = merged_labels["frame_id"].astype(int)
        merged_labels["label"]    = merged_labels["label"].astype(int)

        retrain_dataset_dir = os.path.join(dir_path, "..", "datasets", dataset_name, "train", "retrain_dataset")
        os.makedirs(retrain_dataset_dir, exist_ok=True)

        merged_frames.to_csv(os.path.join(retrain_dataset_dir, file_name[:-4] + "_train_frames.csv"), index=False, header=False)
        merged_labels.to_csv(os.path.join(retrain_dataset_dir, file_name[:-4] + "_train_labels.csv"), index=False)

        print(f"  Original train size      : {train_frames.shape}")
        print(f"  Successful adversarial   : {adv_success.shape}")
        print(f"  Merged size              : {merged_frames.shape}")

        assert merged_frames.shape[0] % 29 == 0

        return retrain_dataset_dir
