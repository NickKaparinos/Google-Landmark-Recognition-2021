"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
from utilities import *
from random import seed
from tensorflow.keras.callbacks import TensorBoard, CSVLogger

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
    path = "path"
    labels = pd.read_csv(
        filepath_or_buffer="path/train.csv")
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
    del labels
    del unique_classes

    batch = training_sequence[0]
    model = build_model((IMG_SIZE, IMG_SIZE, 3))  # x.shape[1:])
    model.fit(x=training_sequence, validation_data=validation_sequence, epochs=4, use_multiprocessing=False,
              callbacks=[tensorboard, csv_logger])

    # Execution Time                tensorboard --logdir "Google Landmark Recognition 2021\logs"
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
