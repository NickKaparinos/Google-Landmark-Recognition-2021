"""
Nick Kaparinos
Google Landmark Recognition 2021
Kaggle Competition
"""

from utilities import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    start = time.perf_counter()
    IMG_SIZE = 200
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
    path = "/home/nickkaparinos/Nikos/"
    labels = pd.read_csv(filepath_or_buffer="/home/nickkaparinos/Nikos/train.csv")
    unique_classes = np.unique(labels.iloc[:, 1])

    # Dataloaders
    validation_dataset = CustomDataset(batch_size=1, data_path=path + '/validation_set',
                                       labels_dataframe_path=path + '/validation_dataframe.csv', IMG_SIZE=IMG_SIZE,
                                       unique_classes=unique_classes, is_validation_dataset=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=64, shuffle=True, num_workers=4,
                                       prefetch_factor=4)

    training_dataset = CustomDataset(batch_size=1, data_path=path + '/training_set',
                                     labels_dataframe_path=path + '/training_dataframe.csv', IMG_SIZE=IMG_SIZE,
                                     unique_classes=unique_classes, is_validation_dataset=False)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=64, shuffle=True, num_workers=4,
                                     prefetch_factor=4)
    del labels
    del unique_classes

    # Model                     # tensorboard --logdir "Google Landmark Recognition 2021\logs"
    model = PytorchTransferModel().to(device)
    learning_rate = 1e-3
    epochs = 3
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(epochs):
        print(f"-----------------Epoch {epoch + 1}-----------------")
        pytorch_train_loop(training_dataloader, model, loss_fn, optimizer, writer, epoch, device)
        pytorch_embedding_test(training_dataloader, validation_dataloader, model, writer, epoch, device)

    # Save model
    torch.save(model.state_dict(), 'model.pth')

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
