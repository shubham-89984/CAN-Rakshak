from train import retrain_model
from get_defense import get_defense_class


def adversarial_retraining(model_path, adversarial_images_path, cfg, adversarial_samples_limit=800):
    print("Adversarial images path :", adversarial_images_path)
    retrainer = get_defense_class(cfg['defense_method'])
    adversarial_dataset = retrainer.make_dataset(
        model_path, adversarial_images_path, cfg, adversarial_samples_limit
    )
    print("Starting adversarial retraining...")
    retrain_model(model_path, adversarial_dataset, cfg)
    print("Adversarial retraining completed.")
