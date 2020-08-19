import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import glob
import os
from PIL import Image


def load_test_batch(args, class_list):
    # Determine the number of classes and number of samples at each class
    fps_train_temp = [args.data_folder + '/' + task_ + '/Test' for task_ in os.listdir(args.data_folder)
                      if int(os.path.split(task_)[-1].split('_')[1]) in args.train_classes]
    number_of_classes = len(fps_train_temp)
    x_data = []
    y_data = []
    for ell in range(number_of_classes):
        files = glob.glob(fps_train_temp[ell] + '/*')
        for k in range(len(files)):
            data = np.asarray(Image.open(files[k]), dtype="float32")
            data = data.reshape(1, 32, 32)
            data = (data - 127.5) / 127.5  # Normalize the images to [-1, 1]
            x_data.append(data)
            y_data.append(ell)
            print(str(k))

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    tensor_x = torch.Tensor(x_data)
    tensor_y = torch.LongTensor(y_data)
    my_dataset = TensorDataset(tensor_x, tensor_y)

    train_ds_ = get_infinite_batches(torch.utils.data.DataLoader(my_dataset, batch_size=args.test_batch_size,
                                                                 shuffle=True, num_workers=args.workers,
                                                                 pin_memory=True))

    return train_ds_


def load_batch(args):
    # Determine the number of classes and number of samples at each class
    fps_train_temp = [args.data_folder + '/' + task_ + '/Train' for task_ in os.listdir(args.data_folder)
                      if int(os.path.split(task_)[-1].split('_')[1]) in args.train_classes]
    number_of_classes = len(fps_train_temp)
    samples_per_class = []
    for ell in range(number_of_classes):
        files = glob.glob(fps_train_temp[ell] + '/*')
        samples_per_class.append(len(files))

    # Load all the samples in a list for each class
    dataset_hor = []
    dataset_lab = []
    for ell in range(number_of_classes):
        files = glob.glob(fps_train_temp[ell] + '/*')
        new = []
        new_lab = []
        for k in range(samples_per_class[ell]):
            data = np.asarray(Image.open(files[k]), dtype="float32")
            data = data.reshape(1, 32, 32)
            data = (data - 127.5) / 127.5  # Normalize the images to [-1, 1]
            new.append(data)
            new_lab.append(ell)
            print(str(k))
        new = np.array(new)
        dataset_hor.append(new)
        new_lab = np.array(new_lab)
        dataset_lab.append(new_lab)

    # First randomly select classes and samples from corresponding classes for each user
    train_ds_ = []
    class_list = []
    sample_list = []
    class_idx = np.arange(number_of_classes)
    for i in range(args.user_number):
        permuted_idx = np.random.permutation(class_idx)
        class_list.append(permuted_idx[:10])
        sample_list_temp = []
        user_data = []
        user_data_labels = []
        for ell in class_list[i]:
            # files = glob.glob(fps_train_temp[ell] + '/*')
            sample_idx = np.arange(samples_per_class[ell])
            permuted_sample_idx = np.random.permutation(sample_idx)
            samples = permuted_sample_idx[:100]
            sample_list_temp.append(samples)
            user_data.extend(dataset_hor[ell][samples])
            user_data_labels.extend(dataset_lab[ell][samples])
        sample_list.append(sample_list_temp)
        x_data = np.array(user_data)
        y_data = np.array(user_data_labels)

        tensor_x = torch.Tensor(x_data)
        tensor_y = torch.LongTensor(y_data)
        my_dataset = TensorDataset(tensor_x, tensor_y)

        train_ds_.append(get_infinite_batches(torch.utils.data.DataLoader(my_dataset,
                                                                          batch_size=args.batch_size,
                                                                          shuffle=True,
                                                                          num_workers=args.workers,
                                                                          pin_memory=True)))

    return train_ds_, class_list


def get_infinite_batches(iterator_obj):
    while True:
        for i, (samples_) in enumerate(iterator_obj):
            yield samples_
