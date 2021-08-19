"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
from utilities import *
from random import seed
from tensorflow.keras.callbacks import TensorBoard

# C:\Users\Nikos\Desktop\Nikos\HMMY\Code\Google Landmark Recognition 2021\Dataset

if __name__ == "__main__":
    start = time.perf_counter()
    IMG_SIZE = 175
    classes = 81313

    # Seeds
    seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # Logs
    LOG_DIR = f"logs/test/{str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))}"
    tensorboard = TensorBoard(log_dir=LOG_DIR)

    # Read labels
    path = "C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset"
    labels = pd.read_csv(
        filepath_or_buffer="C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset/train.csv")
    unique_classes = np.unique(labels.iloc[:, 1])

    # Updated DataLoader
    training_loader = DataLoader(batch_size=64, data_path=path + '/training_set',
                                 labels_dataframe_path=path + '/training_dataframe.csv',
                                 run_validation=True, validation_dataloader=None, IMG_SIZE=IMG_SIZE,
                                 unique_classes=unique_classes)
    del labels
    del unique_classes

    batch = training_loader[0]
    model = returnVGG16((IMG_SIZE, IMG_SIZE, 3))  # x.shape[1:])
    # model.fit(x, y_one_hot)
    # training(data_loader=data_loader, model=model, epochs = 4)

    model.fit(x=training_loader, epochs=4, use_multiprocessing=False)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
    fet = 5
