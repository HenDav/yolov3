import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil


# function that turns XMin, YMin, XMax, YMax coordinates to normalized yolo format
def convert(filename_str, coords):
    image = cv2.imread(filename_str + ".jpg")
    coords[2] -= coords[0]
    coords[3] -= coords[1]
    x_diff = int(coords[2] / 2)
    y_diff = int(coords[3] / 2)
    coords[0] = coords[0] + x_diff
    coords[1] = coords[1] + y_diff
    coords[0] /= int(image.shape[1])
    coords[1] /= int(image.shape[0])
    coords[2] /= int(image.shape[1])
    coords[3] /= int(image.shape[0])
    return coords


ROOT_DIR = os.getcwd()

# step into dataset directory
os.chdir("./data/chest-pics")
files = os.listdir(os.getcwd())
new_files = []
for filename in tqdm(files):
    filename_str = str.split(filename, ".")[0]
    if filename.endswith(".JPG"):
        os.rename(filename, filename_str + ".jpg")
    if filename.endswith(".jpg") and ((filename_str + ".txt") not in files):
        new_files.append(filename_str)
        annotations = []
        with open("../nip_annotations/" + filename_str + ".txt") as f:
            for line in f:
                labels = line.split()
                coords = np.asarray(
                    [float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])])
                coords = convert(filename_str, coords)
                if labels[0] != "0":
                    labels[0], labels[1], labels[2], labels[3], labels[4] = int(labels[0])-1, coords[0], coords[1], coords[2], coords[3]
                    newline = str(labels[0]) + " " + str(labels[1]) + " " + str(labels[2]) + " " + str(
                        labels[3]) + " " + str(labels[4])
                    line = line.replace(line, newline)
                    annotations.append(line)
            f.close()
        with open((filename_str + ".txt"), "w") as outfile:
            for line in annotations:
                outfile.write(line)
                outfile.write("\n")
            outfile.close()

test_fraction = 5   #train test split
files = sorted(os.listdir(os.getcwd()))
test_indecies = np.random.choice(len(new_files), len(new_files)//test_fraction)
if "train" not in files:
    os.mkdir("train")
    os.mkdir("train/images")
    os.mkdir("train/labels")
if "test" not in files:
    os.mkdir("test")
    os.mkdir("test/images")
    os.mkdir("test/labels")
for num, file in enumerate(new_files):
    if num in test_indecies:
        shutil.copy(file + ".jpg", "./test/images")
        shutil.copy(file + ".txt", "./test/labels")
    else:
        shutil.copy(file + ".jpg", "./train/images")
        shutil.copy(file + ".txt", "./train/labels")

