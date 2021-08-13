"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
from utilities import *
from random import seed

# C:\Users\Nikos\Desktop\Nikos\HMMY\Code\Google Landmark Recognition 2021\Dataset

if __name__ == "__main__":
    # Seeds
    seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)


    # Read labels
    path = "C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset"
    labels = dict(pd.read_csv(
        filepath_or_buffer="C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset/train.csv"))

    data_loader = DataLoader(batch_size=64, path=path+'/train')
    data_loader.return_data()

    fet = 5
