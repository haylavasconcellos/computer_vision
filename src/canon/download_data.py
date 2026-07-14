import argparse
import os
import shutil
from pathlib import Path

import kagglehub

from canon.utils import image_utils


def download_kaggle_dataset(dataset_slug, project_name):
    """
    Downloads a Kaggle dataset to the specified local path.

    Args:
        dataset_slug (str): The unique identifier for the dataset
                            (e.g., 'titanic' for 'titanic-dataset' or
                            'datasnaek/youtube-new').
        path (str): The directory where the dataset should be saved.
                    Default is the current directory.
    """

    kaggle_download_path = kagglehub.dataset_download(dataset_slug)

    kaggle_download_path = Path(kaggle_download_path)

    dest_path = image_utils.BASE_DATA_PATH
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
        print(f"Created directory: {dest_path}")

    # Move files to destination
    shutil.move(kaggle_download_path / project_name, dest_path)
    print(f"Dataset downloaded to: {dest_path}")

    # Remove temp files
    os.rmdir(kaggle_download_path)
    os.rmdir(kaggle_download_path.parent)


if __name__ == "__main__":
    datasets = {
        "T1": "eliassantosmartins/mc949-t1",
        "T2": "eliassantosmartins/mc929-t2",
        "T4": "eliassantosmartins/mc949-t4"
    }

    # Configurar argparse para receber argumentos
    parser = argparse.ArgumentParser(description="Download datasets from Kaggle")
    parser.add_argument(
        "--project",
        "-p",
        type=str,
        choices=datasets.keys(),
        help="Trabalho a ser baixado",
        required=True,
    )
    args = parser.parse_args()

    project = args.project
    dataset = datasets[project]
    try:
        download_kaggle_dataset(dataset, project)
    except Exception as e:
        print(e)
