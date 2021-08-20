"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
from utilities import *
from random import seed
from tensorflow.keras.callbacks import TensorBoard, CSVLogger

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
    csv_logger = CSVLogger(filename=LOG_DIR + "/logs.csv")

    file_writer = tf.summary.create_file_writer(LOG_DIR + "/metrics")
    file_writer.set_as_default()

    # Read labels
    path = "C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset"
    labels = pd.read_csv(
        filepath_or_buffer="C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset/train.csv")
    unique_classes = np.unique(labels.iloc[:, 1])

    # Updated DataSequence
    validation_sequence = DataSequence(batch_size=64, data_path=path + '/validation_set',
                                       labels_dataframe_path=path + '/validation_dataframe.csv',
                                       is_validation_sequence=True, IMG_SIZE=IMG_SIZE,
                                       unique_classes=unique_classes)
    training_sequence = DataSequence(batch_size=64, data_path=path + '/training_set',
                                     labels_dataframe_path=path + '/training_dataframe.csv',
                                     is_validation_sequence=False, IMG_SIZE=IMG_SIZE,
                                     unique_classes=unique_classes)
    custom_validation_callback = CustomValidationCallback(validation_sequence=validation_sequence, log_dir=LOG_DIR)
    del labels
    del unique_classes

    batch = training_sequence[0]
    model = returnVGG16((IMG_SIZE, IMG_SIZE, 3))  # x.shape[1:])
    model.fit(x=training_sequence, epochs=5, use_multiprocessing=False,
              callbacks=[tensorboard, csv_logger, custom_validation_callback])
    # fet_pred = model.predict(x=training_sequence)
    # tensorboard --logdir "Google Landmark Recognition 2021\logs"

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
    fet = 5
