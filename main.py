"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
from utilities import *
from random import seed

# C:\Users\Nikos\Desktop\Nikos\HMMY\Code\Google Landmark Recognition 2021\Dataset

if __name__ == "__main__":
    start = time.perf_counter()
    IMAGE_SIZE = 250

    # Seeds
    seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # Read labels
    path = "C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset"
    labels = dict(pd.read_csv(
        filepath_or_buffer="C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset/train.csv").values)

    # Data Loader
    data_loader = DataLoader(batch_size=64, path=path, IMAGE_SIZE=IMAGE_SIZE)
    x,y = data_loader.return_data()

    # names = [b[0] for b in batch]
    # y = [labels[name[:-4]] for name in names]
    print('kassa')
    # y = labels(batch)

    model = returnVGG16(x.shape[1:])

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
    fet = 5
