from helmnet.dataloaders import EllipsesDataset
from torch.utils.data import random_split
from torch import save

if __name__ == "__main__":
    ed = EllipsesDataset()
    ed.make_dataset(num_ellipses=11000, imsize=96)
    ed.sos_maps_to_tensor()

    train_length = 9000
    val_length = 1000
    test_length = 1000

    trainset, valset, testset = random_split(
        dataset=ed,
        lengths=[train_length, val_length, test_length]
    )

    # Save datasets
    save(trainset, 'datasets/splitted_96/trainset.ph')
    save(valset, 'datasets/splitted_96/validation.ph')
    save(testset, 'datasets/splitted_96/testset.ph')