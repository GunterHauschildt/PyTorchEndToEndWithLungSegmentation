import numpy as np
import torch.utils.data as data
import glob
import os
import SimpleITK as sitk
import cv2 as cv
import scipy


def resample_volume(volume, Z, Y, X, mode='interpolate'):
    if mode == 'interpolate':
        volume_p = scipy.ndimage.zoom(
                volume,
                (Z / volume.shape[0],
                 Y / volume.shape[1],
                 X / volume.shape[2]),
                prefilter=False,
                grid_mode=True,
                mode='nearest'
            )
        return volume_p

    elif mode == 'nearest':
        volume_p = np.empty((Z, Y, X), dtype=volume.dtype)
        Zs = [round(z) for z in np.linspace(0, volume.shape[0]-1, num=Z, endpoint=True)]
        for i, z in enumerate(Zs):
            volume_p[i] = cv.resize(volume[z], (X, Y), interpolation=cv.INTER_NEAREST)
        return volume_p


def load_example(volume_file_name, annotation_file_name):

    Z = 48
    X = 96
    Y = 96

    print("Loading Volume:", volume_file_name)
    itk_img = sitk.ReadImage(volume_file_name)
    volume_array = sitk.GetArrayFromImage(itk_img)
    volume_array = volume_array.astype(dtype=np.float32)

    volume_array_n = None
    volume_array_n = cv.normalize(volume_array, volume_array_n, 0.0, 1.0, cv.NORM_MINMAX)
    volume_array_n = resample_volume(volume_array_n, Z, Y, X, 'interpolate')
    volume_array_n = np.expand_dims(volume_array_n, axis=0)

    print("Loading Annotation:", annotation_file_name)
    itk_img = sitk.ReadImage(annotation_file_name)
    annotations = sitk.GetArrayFromImage(itk_img)
    annotations = resample_volume(annotations, Z, Y, X, 'nearest')
    annotations = annotations.astype(dtype=np.int64)

    classes = np.unique(annotations, return_counts=True)

    return volume_array_n, annotations, classes


class LUNA16(data.Dataset):
    @staticmethod
    def find_files(root_dir, volume_dir, annotation_dir):
        annotation_path = os.path.join(root_dir, annotation_dir)
        annotation_files = glob.glob(os.path.join(annotation_path, "*.mhd"))
        annotation_file_names = []
        for annotation_file in annotation_files:
            annotation_file_names.append(os.path.basename(annotation_file)[:-4])

        volume_path = os.path.join(root_dir, volume_dir)
        volume_files = glob.glob(volume_path + "/*.mhd")

        file_names = []
        for volume_file in volume_files:
            volume_file_name = os.path.basename(volume_file)[:-4]
            if volume_file_name not in annotation_file_names:
                continue
            file_names.append(volume_file_name)

        return file_names

    def __init__(self, root_dir, volume_dir, annotation_dir, file_names, max_size=None):

        self._examples = []
        self._classes_np = []
        for file_name in file_names:
            volume, annotations, classes_np = load_example(
                os.path.join(root_dir, volume_dir, file_name + '.mhd'),
                os.path.join(root_dir, annotation_dir, file_name + '.mhd')
            )
            # torch needs a tuple (ie can't use a nicer named container)
            self._examples.append((file_name, volume, annotations))
            self._classes_np.append(classes_np)

            if max_size is not None and len(self._examples) >= max_size:
                break

        self._classes = {}
        for classes_np in self._classes_np:
            for this_class, this_class_count in zip(classes_np[0], classes_np[1]):
                if this_class not in self._classes:
                    self._classes[this_class] = 0
                self._classes[this_class] += this_class_count

    def examples(self):
        return self._examples

    def classes(self):
        return self._classes

    def __getitem__(self, index):
        return self._examples[index]

    def __len__(self):
        return len(self._examples)
