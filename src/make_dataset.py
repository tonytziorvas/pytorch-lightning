import os

import pandas as pd
from torch.utils.data import Dataset


class IntelDataset(Dataset):
    """
    A dataset class for the Intel Natural Scene Classification dataset.

    Attributes:
        data_dir (str): The directory path to the dataset.
        img_paths (List[str]): A list of paths to the images in the dataset.
        labels (List[str]): A list of labels corresponding to the images in the dataset.
        df (pd.DataFrame): A pandas DataFrame containing the image paths and labels.

    Methods:
        __len__ (int): Returns the number of samples in the dataset.
        __getitem__ (pd.Series): Returns a pandas Series containing the image path
          and label at the given index.
        _build_dataset (pd.DataFrame): Builds a pandas DataFrame containing the
          image paths and labels.
    """

    def __init__(self, data_dir: str) -> None:
        """
        Initializes the IntelDataset class.

        Args:
            data_dir (str): The directory path to the dataset.
        """
        self.data_dir = data_dir
        self.img_paths = [
            os.path.join(data_dir, label)
            for label in os.listdir(data_dir)
            if not label.startswith(".")
        ]
        self.labels = [path.split("/")[-1] for path in self.img_paths]
        self.df = self._build_dataset()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.df.shape[0]

    def __getitem__(self, index: int) -> pd.Series:
        """
        Returns a pandas Series containing the image path and label at the given index.

        Args:
            index (int): The index of the sample.

        Returns:
            pd.Series: A pandas Series containing the image path and label.
        """
        return self.df.iat[index]

    def _build_dataset(self) -> pd.DataFrame:
        """
        Builds a pandas DataFrame containing the image paths and labels.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the image paths and labels.
        """
        tmp = {}
        for idx, img_path in enumerate(self.img_paths):
            images = [os.path.join(img_path, img) for img in os.listdir(img_path)]
            for img in images:
                tmp[img] = self.labels[idx]

        return pd.DataFrame(tmp.items(), columns=["img_path", "label"])


if __name__ == "__main__":
    dataset = IntelDataset(data_dir="data/raw/natural_scenes/seg_train")
    print(f"Size: {len(dataset)}")
