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

    # Tensorboard
    LOG_DIR = 'logs/pytorch'
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Seeds
    seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # Read labels
    path = "C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset"
    labels = pd.read_csv(
        filepath_or_buffer="C:/Users/Nikos/Desktop/Nikos/HMMY/Code/Google Landmark Recognition 2021/Dataset/train.csv")
    unique_classes = np.unique(labels.iloc[:, 1])
    # Updated DataSequence
    validation_dataloader = DataLoader(batch_size=64, data_path=path + '/validation_set',
                                       labels_dataframe_path=path + '/validation_dataframe.csv',
                                       is_validation_sequence=True, IMG_SIZE=IMG_SIZE,
                                       unique_classes=unique_classes)
    training_dataloader = DataLoader(batch_size=64, data_path=path + '/training_set',
                                     labels_dataframe_path=path + '/training_dataframe.csv',
                                     is_validation_sequence=False, IMG_SIZE=IMG_SIZE,
                                     unique_classes=unique_classes)
    del labels
    del unique_classes

    # Model
    # batch = training_dataloader[0]
    model = pytorch_model().to(device)  # tensorboard --logdir "Google Landmark Recognition 2021\logs"
    # model = PytorchTransferModel().to(device)
    learning_rate = 1e-3
    epochs = 5
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(epochs):
        print(f"-----------------Epoch {epoch + 1}-----------------")
        pytorch_train_loop(training_dataloader, model, loss_fn, optimizer, writer, epoch, device)
        pytorch_test_loop(validation_dataloader, model, loss_fn, writer, epoch, device)


    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
    fet = 5
