"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
from utilities import *
from random import seed
import pickle

# C:\Users\Nikos\Desktop\Nikos\HMMY\Code\Google Landmark Recognition 2021\Dataset

if __name__ == "__main__":
    start = time.perf_counter()
    IMAGE_SIZE = 150
    classes = 81313

    # Seeds
    seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # Read labels
    path = "C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset"
    labels = dict(pd.read_csv(
        filepath_or_buffer="C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset/train.csv").values)

    # Data Loader
    data_loader = DataLoader(batch_size=120, data_path=path, IMAGE_SIZE=IMAGE_SIZE)
    # data_loader.__getitem__(7)
    # temp = data_loader.return_data()
    # x,y = temp
    # y_one_hot = np.array(tf.one_hot(y, classes))

    model = returnVGG16((IMAGE_SIZE, IMAGE_SIZE, 3))  # x.shape[1:])
    # model.fit(x, y_one_hot)
    # training_epoch(data_loader=data_loader, model=model)

    model.fit(x=data_loader, epochs=4, use_multiprocessing=False)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
    fet = 5
