"""
Nick Kaparinos
Google Landmark Recognition 2021
Kaggle Competition
"""
from utilities import *
from tqdm import tqdm

if __name__ == "__main__":
    start = time.perf_counter()
    IMG_SIZE = 175

    # Seeds
    seed(0)
    np.random.seed(0)

    path = ""
    preprocess_data(path=path, img_size=IMG_SIZE, validation_size=0.25)


    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
