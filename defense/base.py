from common_imports import abc


class BaseDefense(abc.ABC):
    @abc.abstractmethod
    def make_dataset(self, model_path, adversarial_images_path, adversarial_samples_limit=5000):
        """
        Build a merged training dataset combining original frames with
        successful adversarial examples.

        Args:
            model_path:                Path to the trained IDS model (.h5)
            adversarial_images_path:   Path to adversarial .npz file
                                       (must contain 'final_test', 'y_test', 'x_test')
            adversarial_samples_limit: Max number of adversarial samples to include

        Returns:
            str: Path to the retrain dataset directory (passed to retrain_model)
        """
        pass
