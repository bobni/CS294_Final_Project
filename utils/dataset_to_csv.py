import sys
import os
import cv2
import urllib
import tarfile

import imageio
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
import numpy as np
import csv
from io import BytesIO  # "import StringIO" directly in python2
import matplotlib.image as mpimg

cifarClassName = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def download_and_unpack(dst_path):
    # Download CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
    print("Downloading dataset")
    urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'cifar-10-python.tar.gz')
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, dst_path)
    os.remove("cifar-10-python.tar.gz")


def get_dataset_rgb(path):
    # For each training batch read and concatenate into one big numpy array
    # You may find the format of CIFAR-10 in Data Layout section https://www.cs.toronto.edu/~kriz/cifar.html
    # Train Data
    cifarImages = np.empty((0, 3072), dtype=np.uint8)
    cifarLabels = np.empty((0,), dtype=np.uint8)
    for batchNo in range(1, 6):
        dataDict = unpickle(path + '/data_batch_' + str(batchNo))
        cifarImages = np.vstack((cifarImages, dataDict[b'data']))
        cifarLabels = np.hstack((cifarLabels, dataDict[b'labels']))

    train_dataset = cifarImages, cifarLabels

    # Test Data
    dataDict = unpickle(path + '/test_batch')
    cifarTestImages = dataDict[b'data']
    cifarTestLabels = np.array(dataDict[b'labels'])
    test_dataset = cifarTestImages, cifarTestLabels
    return train_dataset, test_dataset


def convert_rgb_to_grayscale(rgb_dataset, quality):
    # Convert all train images to grayscale
    train_dataset_rgb, test_dataset_rgb = rgb_dataset
    train_images_rgb, train_labels = train_dataset_rgb
    test_images_rgb, test_labels = test_dataset_rgb

    cifarImages_rgb_train = []
    for i in range(len(train_images_rgb)):
        image = Image.fromarray(train_images_rgb[i, :].reshape(3, 32, 32).transpose(1, 2, 0))
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=quality)
        image_compressed = np.array(Image.open(buffer))
        cifarImages_rgb_train.append(image_compressed)
    train_images_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in cifarImages_rgb_train])

    cifarImages_rgb_test = []
    for i in range(len(test_images_rgb)):
        image = Image.fromarray(test_images_rgb[i, :].reshape(3, 32, 32).transpose(1, 2, 0))
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=quality)
        image_compressed = np.array(Image.open(buffer))
        cifarImages_rgb_test.append(image_compressed)
    test_images_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in cifarImages_rgb_test])

    return (train_images_gray, train_labels), (test_images_gray, test_labels)


def dataset_gray_to_csv(dataset, dst, classes):
    header = []
    for i in range(32 * 32):
        header.append("Px " + str(i))
    header.append("class")

    images, labels = dataset
    #images = NormalizeData(images)

    i = 0
    with open(dst, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for image, label in zip(images, labels):
            if cifarClassName[label] in classes:
                value = image.flatten()
                value = np.append(value, cifarClassName[label])
                writer.writerow(value)
            i = i + 1
            print('\r' + "Progress: {:4}/{:4} images | {:.2f}% ".format(i, len(images),
                                                                        (float(i) / float(len(images)) * 100.0)),
                  end='')
    print("\n")


def dataset_RGB_to_csv(dataset, dst, quality=100):
    header = []
    for i in range(32 * 32):
        header.append("Px R " + str(i))
        header.append("Px G " + str(i))
        header.append("Px B " + str(i))
    header.append("class")

    i = 0
    image_list = []
    images, labels = dataset

    with open(dst, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for image, label in zip(images, labels):
            image_rgb = image.reshape(3, 32, 32).transpose(1, 2, 0)  # .flatten()
            image_rgb = Image.fromarray(image_rgb)

            buffer = BytesIO()
            image_rgb.save(buffer, "JPEG", quality=quality)
            image_compressed = Image.open(buffer)
            value = np.array(image_compressed).flatten()
            value = np.append(value, cifarClassName[label])
            writer.writerow(value)
            i = i + 1
            print('\r' + "Progress: {:4}/{:4} images | {:.2f}% ".format(i, len(images),
                                                                        (float(i) / float(len(images)) * 100.0)),
                  end='')
    print("\n")


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def jpeg_compress(image, quality):
    buf = BytesIO()
    imageio.imwrite(buf, image, format='jpg', quality=quality)
    s = buf.getbuffer()
    return imageio.imread(s, format='jpg')


def convolve_dataset_2d(gray_dataset, kernel_, padding=0, strides=1):
    kernel = cv2.flip(kernel_, -1)

    train_dataset, test_dataset = gray_dataset
    train_images, train_labels = train_dataset
    test_images, test_labels = test_dataset

    train_images_gray = []
    for image in train_images:
        train_images_gray.append(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT))
    # train_images_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in cifarImages_rgb_train])

    test_images_gray = []
    for image in test_images:
        test_images_gray.append(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT))

    return (train_images_gray, train_labels), (test_images_gray, test_labels)


def generate_compression_overview(dataset, qualitys, mode="gray"):
    images, labels = dataset
    compressed_images = []
    # take first image
    image = images[1]
    if mode == "rgb":
        image_rgb = image.reshape(3, 32, 32).transpose(1, 2, 0)
    im1 = Image.fromarray(image)
    dim = 3
    plt.figure(figsize=(8, 8))
    for i, quality in enumerate(qualitys):
        # here, we create an empty string buffer
        buffer = BytesIO()
        im1.save(buffer, "JPEG", quality=quality)
        size = buffer.tell()
        plt.subplot(dim, dim, i + 1)
        plt.title("q: " + str(quality) + " s: " + str(size) + "bytes")
        plt.axis('off')
        img = mpimg.imread(buffer, format='jpeg')
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        with open("./quality_" + str(quality) + ".jpg", "wb") as file:
            file.write(buffer.getvalue())
        buffer.seek(0)
    plt.savefig("./matrix.jpg")
    plt.show()


def generate_dataset_overview(dataset, quality=20):
    images, labels = dataset
    compressed_images = []
    # take first image
    data = []
    for i in range(0, 10):
        index = np.where(labels == i)[0][0]
        data.append((images[index], labels[index]))

    nrow = 10
    ncol = 3
    # fig = plt.figure()#figsize=(8, 8))
    fig, ax = plt.subplots(nrow, ncol)
    for i, _ in enumerate(data):
        image, label = data[i]

        image_rgb = image.reshape(3, 32, 32).transpose(1, 2, 0)
        image_rgb = Image.fromarray(image_rgb)

        buffer = BytesIO()
        image_rgb.save(buffer, "JPEG", quality=quality)
        size = buffer.tell()

        image_compressed = np.array(Image.open(buffer))
        image_gray = cv2.cvtColor(image_compressed, cv2.COLOR_BGR2GRAY)

        # ax[0, 0].plot((nrow, ncol), (i, 0))
        # plt.title(cifarClassName[label] + " RGB", pad=20)
        ax[i, 0].axis('off')
        ax[i, 0].imshow(image_rgb)

        # plt.subplot2grid((nrow, ncol), (i, 1))

        # plt.title(cifarClassName[label] + " compressed")
        ax[i, 1].axis('off')
        ax[i, 1].imshow(image_compressed)

        # plt.subplot2grid((nrow, ncol), (i, 2))

        # plt.title(cifarClassName[label] + " gray")
        ax[i, 2].axis('off')
        ax[i, 2].imshow(image_gray, cmap='gray', vmin=0, vmax=255)

        buffer.seek(0)
    plt.savefig("./matrix.jpg")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_path = "../datasets/"
    cifar_path = dataset_path + "cifar-10-batches-py"

    if not os.path.isdir(cifar_path):
        download_and_unpack(dst_path=cifar_path)

    dataset_rgb = get_dataset_rgb(cifar_path)  # (train_dataset, test_dataset)
    dataset_RGB_train, dataset_RGB_test = dataset_rgb

    # generate_dataset_overview(dataset_RGB_train)
    # exit()
    # quality_list = [1, 5, 10, 15, 20, 25, 50, 75, 100]
    quality_list = [20]

    # generate_compression_overview(dataset_RGB_train, quality_list, mode="rgb")

    # for quality in quality_list:
    #     path_train = dataset_path + "cifar10_RGB_quality_" + str(quality) + "_train.csv"
    #     path_test = dataset_path + "cifar10_RGB_quality_" + str(quality) + "_test.csv"
    #     if not os.path.isfile(path_train):
    #         dataset_RGB_to_csv(dataset=dataset_RGB_train, dst=path_train, quality=quality)
    #     if not os.path.isfile(path_test):
    #         dataset_RGB_to_csv(dataset=dataset_RGB_test, dst=path_test, quality=quality)
    #
    # exit()

    # generate_compression_overview(dataset_gray_train, quality_list, mode="gray")
    # exit()
    classes = ['cat', 'dog']

    for quality in quality_list:
        dataset_gray_train, dataset_gray_test = convert_rgb_to_grayscale(dataset_rgb, quality=quality)
        path_train = dataset_path + "gray_cat_dog_q_" + str(quality) + "_train.csv"
        path_test = dataset_path + "gray_cat_dog_q_" + str(quality) + "_test.csv"
        if not os.path.isfile(path_train):
            dataset_gray_to_csv(dataset=dataset_gray_train, dst=path_train, classes=classes)
        if not os.path.isfile(path_test):
            dataset_gray_to_csv(dataset=dataset_gray_test, dst=path_test, classes=classes)

    # if not os.path.isfile(dataset_path + "cifar10_gray_train.csv"):
    #     dataset_gray_to_csv(dataset=dataset_gray_train, dst=dataset_path + "cifar10_gray_train.csv")
    #
    # if not os.path.isfile(dataset_path + "cifar10_gray_test.csv"):
    #     dataset_gray_to_csv(dataset=dataset_gray_test, dst=dataset_path + "cifar10_gray_test.csv")
