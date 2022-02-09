import cv2
import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import Dataset
from tqdm import trange


def get_dataset(
        dataset_path: str, source_location="cuda:7", destination="cpu"
) -> Dataset:
    """Loads a torch dataset and maps it to arbitrary locations

    Args:
        dataset_path (str): Path of the dataset. It must be a .ph file
        source_location (str, optional): On which device the dataset was located. Defaults to "cuda:7".
        destination (str, optional): On which device the dataset must be mapped to. Defaults to "cpu".

    Returns:
        torch.Dataset
    """
    # Preparing dataset
    trainset = torch.load(dataset_path, map_location={source_location: destination})
    return trainset


class EllipsesDataset(Dataset):
    """Dataset of oversimplified skulls."""

    def __init__(self):
        self._all_sos = []
        self.all_sos_numpy = []

    @property
    def all_sos(self):
        if len(self._all_sos) == 0 and len(self.all_sos_numpy) == 0:
            print("You probably didn't call method `make_dataset`.")
            return []
        elif len(self._all_sos) == 0:
            print("You probably didn't call method `sos_maps_to_tensor`.")
            return []
        return self._all_sos

    def make_dataset(self, num_ellipses=5000, imsize=128):
        """Generates a dataset of oversimplified skulls.

        Args:
            num_ellipses (int, optional): How many maps to make. Defaults to 5000.
            imsize (int, optional): Size of the speed of sound map. Possibly
                a power of two. The map is squared. Defaults to 128.
        """
        all_sos_maps = []
        for _ in trange(num_ellipses):
            all_sos_maps.append(self._make_ellipsoid(imsize))
        self.all_sos_numpy = np.stack(all_sos_maps, axis=0)

    def load_dataset(self, filepath="data/ellipses.npy"):
        """Loads a dataset from a `npy` file

        Args:
            filepath (str, optional): Relative file path. Defaults to "data/ellipses.npy".
        """
        all_sos = np.load(filepath)
        self.all_sos_numpy = np.array(all_sos, np.float32)

    def save_dataset(self, filepath: str):
        """Saves a dataset as an `npy` file.

        Args:
            filepath (str): Path to save the file. Should start from the
                folder `data` to avoid confusion.
        """
        np.save(filepath, self.all_sos_numpy)

    def save_for_matlab(self, name):
        savemat("datasets/" + name, {"speeds_of_sound": self._all_sos.numpy()})

    def sos_maps_to_tensor(self):
        """Moves the maps to a cuda tensor and takes care of some shaping"""
        self._all_sos = torch.from_numpy(self.all_sos_numpy).unsqueeze(1).float()

    @staticmethod
    def _make_ellipsoid(imsize: int = 128,
                        avg_thickness: float = 2,
                        std_thickness: float = 8,
                        background_sos: float = 1.0,
                        minimal_skull_sos_boost: float = 0.5,
                        maximal_random_skull_boost: float = 0.5,
                        avg_amplitudes_tuple: tuple[float] = (1.0, 0.0, 0.0, 0.0),
                        std_amplitudes_tuple: tuple[float] = (0.1, 0.05, 0.025, 0.01),
                        std_phase_value: float = np.pi / 16,
                        avg_phase_value: float = 0):
        """
        Internal method to make an ellipsoid speed of sound map.

        Args:
            imsize (int, optional): Size of the image. Defaults to 128.
            avg_thickness (float, optional): average of the thickness
            std_thickness (float, optional): std of the thickness
            background_sos (float, optional): background speed of sound
            minimal_skull_sos_boost (float, optional): minimal difference between background sos and skull sos
            maximal_random_skull_boost (float, optional): maximal value randomly added to background sos
                to compute skull sos
            avg_amplitudes_tuple (tuple of floats, optional): contains average of amplitudes
                summing up to generate harmonic
            std_amplitudes_tuple (tuple of floats, optional): contains std of amplitudes
                summing up to generate harmonic
            std_phase_value (float, optional): std of the phase
            avg_phase_value (float, optional): average of the phase


        Returns:
            np.array: The speed of sound map with a random ellipsoid.
        """
        t = np.linspace(0, 2 * np.pi, num=360, endpoint=True)

        # Distribution parameters

        avg_amplitudes = np.array(avg_amplitudes_tuple)
        std_amplitudes = np.array(std_amplitudes_tuple)
        harmonics_count = len(avg_amplitudes)
        avg_phase = np.array([avg_phase_value] * harmonics_count)
        std_phase = np.array([std_phase_value] * harmonics_count)

        # Generate sample
        a_x = (avg_amplitudes + np.random.randn(4) * std_amplitudes)
        a_y = (avg_amplitudes + np.random.randn(4) * std_amplitudes)

        ph_x = (avg_phase + np.random.randn(4) * std_phase)
        ph_y = (avg_phase + np.random.randn(4) * std_phase)

        x = 0.0
        y = 0.0
        for i in range(harmonics_count):
            x = x + np.sin(t * (i + 1) + ph_x[i]) * a_x[i]
            y = y + np.cos(t * (i + 1) + ph_y[i]) * a_y[i]
        x = (x + 2) / harmonics_count
        y = (y + 2) / harmonics_count

        # Transform into image
        thickness = int(avg_thickness + np.random.rand(1, ) * std_thickness)
        img = np.zeros((imsize, imsize, 3), dtype="uint8")

        x = x * imsize
        y = y * imsize
        pts = np.expand_dims(np.array([x, y], np.int32).T, axis=0)

        cv2.polylines(img, [pts], True, (1, 0, 0), thickness=thickness)

        # Fixing speed of sound
        random_skull_sos_boost = np.random.rand(1) * maximal_random_skull_boost
        rand_amplitude = (random_skull_sos_boost + minimal_skull_sos_boost)
        img = np.array(img[:, :, 0], np.float32) * rand_amplitude
        sos = background_sos + img

        return sos

    def __len__(self):
        return len(self._all_sos)

    def __getitem__(self, idx):
        return self._all_sos[idx]
