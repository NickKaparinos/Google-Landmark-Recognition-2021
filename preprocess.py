"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
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
    tf.random.set_seed(0)

    path = "C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset"
    preprocess_data(path=path, img_size=IMG_SIZE, validation_size=0.25)


    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
