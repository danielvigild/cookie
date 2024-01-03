import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    path = "/Users/sorenbendtsen/Documents/GitHub/dtu_mlops/data/corruptmnist/"
    train_images, train_labels = [], []
    for i in range(1, 6):
        train_images.append(torch.load(path + "train_images_" + str(i) + ".pt"))
        train_labels.append(torch.load(path + "train_target_" + str(i) + ".pt"))

    test_images = torch.load(path + "test_images.pt")
    test_labels = torch.load(path + "test_target.pt")
    
    # stack the tensors
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)
    
    # normalize the images with mean 0 and std 1
    mean = train_images.mean()
    std = train_images.std()
    for i in range(len(train_images)):
        train_images[i] = (train_images[i] - mean) / std

    # convert to torch tensors
    train = torch.utils.data.TensorDataset(train_images, train_labels)
    test = torch.utils.data.TensorDataset(test_images, test_labels)

    # create dataloaders
    train = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    return train, test

train, test = mnist()

    # save the training and test data to processed
    torch.save(train, "/Users/sorenbendtsen/Documents/GitHub/mlops/day2_soren_cookiecutter_project/data/processed/train.pt")
    torch.save(test, "/Users/sorenbendtsen/Documents/GitHub/mlops/day2_soren_cookiecutter_project/data/processed/test.pt")




