import numpy as np
import cv2 as cv
import os
import skimage
import math
import random
import napari


class Colors:
    _colors_bgr0255 = None
    _colors_bgr01 = None

    @staticmethod
    def _init(num: int = 100):
        if Colors._colors_bgr0255 is None:
            # random.seed(4)
            # the cv and napari colors only seem to match
            # if very bright
            Colors._colors_bgr0255 = [
                (random.randint(0, 255),
                 random.randint(0, 255),
                 random.randint(0, 255))
                for _ in range(num)]
            Colors._colors_bgr01 = []
            for i in range(len(Colors._colors_bgr0255)):
                Colors._colors_bgr01.append((
                    Colors._colors_bgr0255[i][0] / 255.,
                    Colors._colors_bgr0255[i][1] / 255.,
                    Colors._colors_bgr0255[i][2] / 255.))

    @staticmethod
    def colors_bgr0255(num: int = 100):
        if Colors._colors_bgr0255 is None:
            Colors._init(num)
        return Colors._colors_bgr0255

    @staticmethod
    def color_bgr0255(n: int) -> tuple[int, int, int]:
        if Colors._colors_bgr0255 is None:
            Colors._init()
        return Colors._colors_bgr0255[n % len(Colors._colors_bgr0255)]

    @staticmethod
    def color_bgr01(n: int) -> tuple[float, float, float]:
        if Colors._colors_bgr01 is None:
            Colors._init()
        return Colors._colors_bgr01[n % len(Colors._colors_bgr01)]


class DrawLabelled3dImageAsGrid:
    def __init__(self, image_shape, scale=1.0, grid=(7, None)):
        self._image_shape = image_shape
        self._scale = scale
        self._grid = (grid[0], math.ceil(image_shape[0] / grid[0]))
        self._image_grid = np.zeros(
            (image_shape[1] * self._grid[0], image_shape[2] * self._grid[1], 3),
            dtype=np.uint8)

        w = round(self._image_grid.shape[1] * self._scale)
        h = round(self._image_grid.shape[0] * self._scale)
        self._size_xy = w, h
        self._normalize = True

    def size_xy(self) -> tuple[int, int]:
        return self._size_xy

    def grid(self):
        return self._grid

    def image_grid(self):
        return self._image_grid

    def draw(self, image_3d, image_labels_3d, labels):

        image_3d = image_3d.astype(dtype=np.float32)

        if self._normalize:
            image_3d_n = np.zeros_like(image_3d)
            image_3d = cv.normalize(image_3d, image_3d_n, 0.0, 1.0, cv.NORM_MINMAX)

        image_3d *= 255.0
        image_3d = image_3d_n.astype(dtype=np.uint8)

        for i in range(image_3d.shape[0]):
            this_grid_r = i // self._grid[1]
            this_grid_c = i % self._grid[1]
            img_draw = np.expand_dims(image_3d[i], axis=2)
            img_draw = cv.cvtColor(img_draw, cv.COLOR_GRAY2BGR)

            # segmentations = np.unique(image_labels_3d[i])
            for label in labels:
                if label == 0:
                    continue
                mask = np.where(image_labels_3d[i] == label, 1, 0).astype(dtype=np.uint8)
                mask = np.expand_dims(mask, axis=2)
                contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                img_draw = cv.drawContours(img_draw, contours, -1,
                                           Colors.color_bgr0255(label), 2)

            grid_y0 = this_grid_r * image_3d.shape[1]
            grid_y1 = grid_y0 + image_3d.shape[1]
            grid_x0 = this_grid_c * image_3d.shape[2]
            grid_x1 = grid_x0 + image_3d.shape[2]
            img_draw = np.flip(img_draw, axis=1)

            self._image_grid[grid_y0:grid_y1, grid_x0:grid_x1, :] = img_draw

        return cv.resize(self._image_grid, self._size_xy)


class DrawLabelled3dImageAsVolume:

    def __init__(self, volume_shape, scale_2d=1.0):
        self._volume_shape = volume_shape
        self._rotate = [75, 0, 200]
        self._scale_3d = [1.5, 1.0, 1.0]
        self._scale_2d = scale_2d
        self._size_xy = None
        m = np.zeros((self._volume_shape[0], self._volume_shape[1], self._volume_shape[2], 3),
                     dtype=np.uint8)
        m_3d = self.draw(m, None, [], False)
        self._image_shape = m_3d.shape[:2]

    def size_xy(self) -> tuple[int, int]:
        return self._size_xy

    def draw(self, image_3d, image_labels_3d, valid_labels, open):
        viewer = napari.view_image(
            image_3d, scale=self._scale_3d, ndisplay=3, rotate=self._rotate,
            colormap='gray_r', name="volume", title="3D segmentation"
        )

        if image_labels_3d is not None:
            labels = []
            for i in valid_labels:
                if open:
                    labels.append(skimage.morphology.binary_opening(
                                np.where(image_labels_3d == i, 1, 0),
                                footprint=skimage.morphology.ball(1))
                    )
                else:
                    labels.append(np.where(image_labels_3d == i, 1, 0))

            for l, label in enumerate(labels):
                viewer.add_labels(label,
                                  scale=self._scale_3d,
                                  rotate=self._rotate,
                                  name=f"{l}",
                                  opacity=.75,
                                  num_colors=2,
                                  color={
                                      0: None,
                                      1: Colors.color_bgr01(l)
                                  },
                                  visible=True)

        # TODO USE TEMP FILES
        screenshots_root = "./"
        screenshots_folder = os.path.join(screenshots_root)
        os.makedirs(screenshots_folder, exist_ok=True)
        screenshot_file = os.path.join(screenshots_folder, "3d.png")
        m = viewer.screenshot(screenshot_file)
        m = cv.cvtColor(m, cv.COLOR_BGRA2BGR)
        viewer.close()
        # cv.imshow("DRAW", m)
        # cv.waitKey()
        if self._size_xy is None:
            self._size_xy = (round(m.shape[1] * self._scale_2d),
                             round(m.shape[0] * self._scale_2d))
        m = cv.resize(m, self._size_xy)
        return m


class VideoWriter:
    def __init__(self, volume_size_xy, grid_size_xy):
        # self._epochs = epochs
        # self._batch_size = batch_size

        self._ribbon_1_height = 32
        self._ribbon_2_height = 32
        self._size_xy = (
            max(volume_size_xy[0], 2 * grid_size_xy[0]),
            volume_size_xy[1] + grid_size_xy[1] +
            self._ribbon_1_height + self._ribbon_2_height
        )
        self._video = np.zeros((self._size_xy[1], self._size_xy[0], 3)).astype(np.uint8)
        self._ribbon_1_x0y0x1y1 = ((0, 0), (self._video.shape[1], self._ribbon_1_height,))
        self._volume_lhs_x0y0x1y1 = ((0, self._ribbon_1_height),
                                     (volume_size_xy[0], self._ribbon_1_height + volume_size_xy[1]))
        self._volume_rhs_x0y0x1y1 = ((volume_size_xy[0], self._ribbon_1_height),
                                     (volume_size_xy[0]+volume_size_xy[0],
                                      self._ribbon_1_height + volume_size_xy[1]))
        self._ribbon_2_x0y0x1y1 = ((0, self._ribbon_1_height + volume_size_xy[1]),
                                   (self._video.shape[1],
                                    self._ribbon_1_height + volume_size_xy[1] +
                                    self._ribbon_2_height))
        self._grid_lhs_x0y0x1y1 = ((0, self._ribbon_1_height + volume_size_xy[1] +
                                    self._ribbon_2_height),
                                   (grid_size_xy[0], self._video.shape[0]))
        self._grid_rhs_x0y0x1y1 = ((grid_size_xy[0],
                                    self._ribbon_1_height + volume_size_xy[1] + self._ribbon_2_height),
                                   (grid_size_xy[0]+grid_size_xy[0], self._video.shape[0]))

        self._video_writer = cv.VideoWriter("unet_time_lapse.mp4",
                                            cv.VideoWriter.fourcc(*'mp4v'),
                                            10,
                                            self._size_xy)

    def put_volume_lhs(self, volume):
        (l, t), (r, b) = self._volume_lhs_x0y0x1y1
        self._video[t:b, l:r] = volume

    def put_volume_rhs(self, volume):
        (l, t), (r, b) = self._volume_rhs_x0y0x1y1
        self._video[t:b, l:r] = volume

    def put_grid_lhs(self, grid):
        (l, t), (r, b) = self._grid_lhs_x0y0x1y1
        self._video[t:b, l:r] = grid

    def put_grid_rhs(self, grid):
        (l, t), (r, b) = self._grid_rhs_x0y0x1y1
        self._video[t:b, l:r] = grid

    def _put_ribbon(self, rect, text):
        (l, t), (r, b) = rect
        self._video[t:b, l:r] = 0
        cv.putText(self._video[t:b, l:r], text, (10, 16),
                   cv.FONT_HERSHEY_PLAIN, 1.0,  (255, 255, 255))

    def put_ribbon_1(self, text):
        self._put_ribbon(self._ribbon_1_x0y0x1y1, text)

    def put_ribbon_2(self, text):
        self._put_ribbon(self._ribbon_2_x0y0x1y1, text)

    def show(self, wait=0):
        cv.imshow("UNET Time-Lapse", self._video)
        self._video_writer.write(self._video)
        cv.waitKey(wait)


def main():
    np_root = "C:/Users/gunte/OneDrive/Desktop/Code/Luna16/np"

    input_shape = [48, 96, 96]
    volume_draw = DrawLabelled3dImageAsVolume(input_shape, .40)
    grid_draw = DrawLabelled3dImageAsGrid(input_shape, .75)
    video_writer = VideoWriter(volume_draw.size_xy(), grid_draw.size_xy())

    num_epochs = 12
    s = 0
    for epoch in range(0, num_epochs):

        def image_target_file_name(b, x, is_read):
            fn = f"{b:04}" f"_" + f"{x:04}"
            if is_read:
                fn += ".npy"
            return fn

        def output_file_name(epoch, b, x, is_read):
            fn = f"{epoch:04}" + f"_" + f"{b:04}" + f"_" + f"{x:04}"
            if is_read:
                fn += ".npy"
            return fn

        def load_files(folder, epoch, b, x):
            image_target_fn = image_target_file_name(b, x, True)
            np_image_file = os.path.join(np_root, folder + "_images", image_target_fn)
            np_target_file = os.path.join(np_root, folder + "_targets", image_target_fn)
            output_fn = output_file_name(epoch, b, x, True)
            np_output_file = os.path.join(np_root, folder + "_outputs", output_fn)

            if not (os.path.isfile(np_image_file) and
                    os.path.isfile(np_target_file) and
                    os.path.isfile(np_output_file)):
                return None, None, None

            image = np.load(np_image_file)
            image = np.squeeze(image, axis=0)
            output = np.load(np_output_file)
            target = np.load(np_target_file)
            return image, target, output

        # first validation. draw as 3d volume. stick to just the 0th image
        # in the 0ths batch. That way we can draw one time-lapse video
        # as it improves.
        x = 0
        b = 0
        image, target, output = load_files("valid", epoch, x, b)
        if image is None or target is None or output is None:
            continue
        v_output = volume_draw.draw(image, output, [3, 4, 5], epoch != 0)
        v_target = volume_draw.draw(image, target, [3, 4, 5], epoch != 0)
        video_writer.put_volume_lhs(v_target)
        video_writer.put_volume_rhs(v_output)
        video_writer.put_ribbon_1(
            (f"Epoch {epoch: 2}: Validate Target & Neural Network Output "
             f"(shown as volume)"))

        if epoch != 0:
            video_writer.show(100)

        # for training images, show every batch for every epoch but just the first sample
        step = 5, 4
        for b in range(0, 80, step[0]):
            for x in range(1, 8, step[1]):
                image, target, output = load_files("train", epoch, x, b)
                if image is None or target is None or output is None:
                    continue
                g_output = grid_draw.draw(image, output, [3, 4, 5])
                g_target = grid_draw.draw(image, target, [3, 4, 5])
                video_writer.put_grid_lhs(g_target)
                video_writer.put_grid_rhs(g_output)
                video_writer.put_ribbon_2(
                    (f"Epoch {epoch: 2}, Total Training Samples: {(s:=s+(step[0]*step[1])): 4} "
                     "Training Target & Neural Network Output "
                     "(shown as slices)")
                )
                video_writer.show(100)


if __name__ == "__main__":
    main()
