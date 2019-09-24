"""A GUI for interacting with a trained descriptor network.
"""

import argparse
import os
import sys
import glob
import pickle

import torch
import torch.nn as nn
import numpy as np
import matplotlib.cm as cm

from form2fit import config
from form2fit.code.ml.models import *
from form2fit.code.ml.dataloader import get_corr_loader
from form2fit.code.utils import ml, misc

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Debugger(QDialog):
    """A PyQt5 GUI for debugging a descriptor network.
    """

    USE_CUDA = True
    WINDOW_WIDTH = 1500
    WINDOW_HEIGHT = 1000
    WINDOW_TITLE = "Debug Descriptor Network"

    def __init__(self, args):
        super().__init__()

        self._foldername = args.foldername
        self._dtype = args.dtype
        self._num_desc = args.num_desc
        self._background_subtract = args.background_subtract
        self._augment = args.augment
        self._num_channels = args.num_channels
        self._init_loader_and_network()
        self._reset()
        self._init_UI()
        self.show()

    def _init_loader_and_network(self):
        """Initializes the data loader and network.
        """
        self._dev = torch.device("cuda" if Debugger.USE_CUDA and torch.cuda.is_available() else "cpu")
        self._data = get_corr_loader(
            self._foldername,
            batch_size=1,
            sample_ratio=1,
            dtype=self._dtype,
            shuffle=False,
            num_workers=0,
            augment=self._augment,
            num_rotations=20,
            background_subtract=config.BACKGROUND_SUBTRACT[self._foldername],
            num_channels=self._num_channels,
        )
        self._net = CorrespondenceNet(self._num_channels, self._num_desc, 20).to(self._dev)
        self._net.eval()
        stats = self._data.dataset.stats
        self._color_mean = stats[0][0]
        self._color_std = stats[0][1]
        self._resolve_data_dims()

    def _resolve_data_dims(self):
        """Reads the image dimensions from the data loader.
        """
        x, _, _ = next(iter(self._data))
        self._h, self._w = x.shape[2:]
        self._c = 3
        self._zeros = np.zeros((self._h, self._w, self._c), dtype="uint8")
        self.xs = None
        self.xt = None

    def _reset(self):
        """Resets the GUI.
        """

        def _he_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")

        self._is_switch = False
        self._pair_idx = 0
        self._dloader = iter(self._data)
        self._get_network_names()
        self._net.apply(_he_init)

    def _get_network_names(self):
        """Reads all saved model weights.
        """
        self.weights_dir = os.path.join(config.weights_dir, "matching")
        filenames = glob.glob(os.path.join(self.weights_dir, "*.tar"))
        self._model_names = [os.path.basename(x).split(".")[0] for x in filenames]

    def _load_selected_network(self, name):
        """Loads a trained network.
        """
        if name:
            self._model_name = name
            state_dict = torch.load(os.path.join(self.weights_dir, name + ".tar"), map_location=self._dev)
            self._net.load_state_dict(state_dict['model_state'])
            self._set_prediction_text("{} was loaded...".format(name))

    def _init_UI(self):
        """Initializes the UI.
        """
        self.setWindowTitle(Debugger.WINDOW_TITLE)
        # self.setFixedSize(Debugger.WINDOW_WIDTH, Debugger.WINDOW_HEIGHT)

        self._create_menu()
        self._create_main()
        self._create_progress()

        self._all_layout = QVBoxLayout(self)
        self._all_layout.addLayout(self._menu_layout)
        self._all_layout.addLayout(self._main_layout)
        self._all_layout.addLayout(self._progress_layout)

    def _create_menu(self):
        """Creates the top horizontal menu bar.
        """
        # buttons
        next_button = QPushButton("Next Pair", self)
        next_button.clicked.connect(self._next_click)
        reset_button = QPushButton("Reset", self)
        reset_button.clicked.connect(self._reset_click)
        sample_button = QPushButton("Sample", self)
        sample_button.clicked.connect(self._sample_click)
        colorize_button = QPushButton("Rotation Error", self)
        colorize_button.clicked.connect(self._colorize_click)
        self._switch_button = QPushButton("View RGB", self)
        self._switch_button.clicked.connect(self._switch_click)

        # boxes
        self._is_correct_box = QLabel(self)
        self._networks_box = QComboBox(self)
        self._networks_box.addItems([""] + self._model_names)
        self._networks_box.activated[str].connect(self._load_selected_network)
        self._networks_box_label = QLabel("Network Name", self)
        self._networks_box_label.setBuddy(self._networks_box)

        # add to layout
        self._menu_layout = QHBoxLayout()
        self._menu_layout.addWidget(self._networks_box_label)
        self._menu_layout.addWidget(self._networks_box)
        self._menu_layout.addWidget(next_button)
        self._menu_layout.addWidget(sample_button)
        self._menu_layout.addWidget(colorize_button)
        self._menu_layout.addWidget(self._is_correct_box)
        self._menu_layout.addStretch(1)
        self._menu_layout.addWidget(self._switch_button)
        self._menu_layout.addWidget(reset_button)

    def _create_main(self):
        """Creates the main layout.
        """
        vbox_left = QVBoxLayout()
        grid_right = QGridLayout()

        self._target_widget = QLabel(self)
        self._source_widget = QLabel(self)

        self._grid_widgets = [QLabel(self) for _ in range(20)]
        self._draw_target(init=True)
        self._draw_source(init=True)
        vbox_left.addWidget(self._target_widget)
        vbox_left.addWidget(self._source_widget)
        self._target_widget.mousePressEvent = self._get_mouse_pos
        self._draw_rotations(init=True)
        for col in range(5):
            for row in range(4):
                grid_right.addWidget(self._grid_widgets[col * 4 + row], col, row)

        self._main_layout = QHBoxLayout()
        self._main_layout.addLayout(vbox_left)
        self._main_layout.addLayout(grid_right)

    def _create_progress(self):
        """A progress bar for the data loader.
        """
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, len(self._dloader))
        self._progress_bar.setValue(0)
        self._progress_layout = QHBoxLayout()
        self._progress_layout.addWidget(self._progress_bar)
        self._advance_progress_bar()

    def _draw_target(self, uv=None, init=False):
        img_target = self._zeros.copy() if init else self._xt_np.copy()
        if uv is not None:
            img_target[uv[0] - 1 : uv[0] + 1, uv[1] - 1 : uv[1] + 1] = [255, 0, 0]
        self._target_img = QImage(
            img_target.data, self._w, self._h, self._c * self._w, QImage.Format_RGB888
        )
        self._target_pixmap = QPixmap.fromImage(self._target_img)
        self._target_widget.setPixmap(self._target_pixmap)
        self._target_widget.setScaledContents(True)

    def _draw_source(self, uvs=None, init=False):
        if uvs is None:
            img_source = self._zeros.copy() if init else self._xs_np.copy()
        else:
            img_source = self._xt_np.copy()
            colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
            color_names = ["green", "blue", "red"]
            for i in range(3):
                mask = np.where(uvs[:, 2] == i)[0]
                idxs = uvs[mask]
                img_source[idxs[:, 0], idxs[:, 1]] = colors[i]
        self._source_img = QImage(
            img_source.data, self._w, self._h, self._c * self._w, QImage.Format_RGB888
        )
        self._source_pixmap = QPixmap.fromImage(self._source_img)
        self._source_widget.setPixmap(self._source_pixmap)
        self._source_widget.setScaledContents(True)

    def _draw_rotations(self, init=False, heatmap=True):
        def _hist_eq(img):
            from skimage import exposure

            img_cdf, bin_centers = exposure.cumulative_distribution(img)
            return np.interp(img, bin_centers, img_cdf)

        for col in range(5):
            for row in range(4):
                offset = col * 4 + row
                if init:
                    img = self._zeros.copy()
                else:
                    if heatmap:
                        img = self.heatmaps[offset].copy()
                        img = img / img.max()
                        img = _hist_eq(img)
                        img = np.uint8(cm.viridis(img) * 255)[..., :3]
                        img = img.copy()
                    else:
                        img = misc.rotate_img(self._xs_np, -(360 / 20) * offset, center=(self.center[1], self.center[0]))
                        img = img.copy()
                    if offset == self._uv[-1]:
                        img[
                            self._uv[0] - 1 : self._uv[0] + 1,
                            self._uv[1] - 1 : self._uv[1] + 1,
                        ] = [255, 0, 0]
                        self._add_border_clr(img, [255, 0, 0])
                    if offset == self.best_rot_idx:
                        self._add_border_clr(img, [0, 255, 0])
                self._img = QImage(
                    img.data, self._w, self._h, self._c * self._w, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(self._img)
                self._grid_widgets[offset].setPixmap(pixmap)
                self._grid_widgets[offset].setScaledContents(True)

    def _switch_click(self):
        if not self._is_switch:
            self._switch_button.setText("Heatmap View")
            self._is_switch = True
            self._draw_rotations(heatmap=False)
        else:
            self._switch_button.setText("RGB View")
            self._is_switch = False
            self._draw_rotations(heatmap=True)

    def _next_click(self):
        if self._pair_idx == len(self._dloader):
            self.close()
        else:
            self._get_next_data()
            self._draw_target()
            self._draw_source()
            self._draw_rotations(init=True)
            self._advance_progress_bar()

    def _reset_click(self):
        self._reset()
        self._networks_box.setCurrentIndex(0)
        self._draw_target(init=True)
        self._draw_source(init=True)
        self._draw_rotations(init=True)
        self._advance_progress_bar()

    def _colorize_click(self):
        filename = os.path.join(
            config.rot_stats_dir,
            self._model_name,
            self._dtype,
            str(self._pair_idx - 1),
            "rot_color.npy",
        )
        pixel_colors = np.load(filename)
        self._draw_source(pixel_colors)

    def _set_prediction_text(self, msg):
        self._is_correct_box.setText(msg)

    def _sample_click(self):
        if self._pair_idx > 0:
            self._forward_network()
            rand_idx = np.random.choice(np.arange(len(self.target_pixel_idxs)))
            u_rand, v_rand = self.target_pixel_idxs[rand_idx]
            self._draw_target([u_rand, v_rand])
            u_s, v_s = self.source_pixel_idxs[rand_idx]
            target_vector = self.out_t[:, :, u_rand, v_rand]
            outs_flat = self.outs.view(self.outs.shape[0], self.outs.shape[1], -1)
            target_vector_flat = target_vector.unsqueeze_(2).repeat(
                (outs_flat.shape[0], 1, outs_flat.shape[2])
            )
            diff = outs_flat - target_vector_flat
            dist = diff.pow(2).sum(1).sqrt()
            self.heatmaps = dist.view(dist.shape[0], self._h, self._w).cpu().numpy()
            predicted_best_idx = dist.min(dim=1)[0].argmin()
            is_correct = predicted_best_idx == self.best_rot_idx
            msg = "Correct!" if is_correct else "Wrong!"
            self._set_prediction_text(msg)
            min_val = self.heatmaps[predicted_best_idx].argmin()
            u_min, v_min = misc.make2d(min_val, self._w)
            self._uv = [u_min, v_min, predicted_best_idx]
            self._draw_rotations(heatmap=not self._is_switch)
        else:
            print("[!] You must first click next to load a data sample.")


    def _get_mouse_pos(self, event):
        v = event.pos().x()
        u = event.pos().y()
        u = int(u * (self._h / self._target_widget.height()))
        v = int(v * (self._w / self._target_widget.width()))
        uv = [u, v]
        if self.xs is not None and self.xt is not None:
            self._forward_network()
            row_idx = np.where((self.target_pixel_idxs == uv).all(axis=1))[0]
            if row_idx.size != 0:
                row_idx = row_idx[0]
                self._draw_target(uv)
                u_s, v_s = self.source_pixel_idxs[row_idx]
                target_vector = self.out_t[:, :, uv[0], uv[1]]
                outs_flat = self.outs.view(self.outs.shape[0], self.outs.shape[1], -1)
                target_vector_flat = target_vector.unsqueeze_(2).repeat(
                    (outs_flat.shape[0], 1, outs_flat.shape[2])
                )
                diff = outs_flat - target_vector_flat
                dist = diff.pow(2).sum(1).sqrt()
                self.heatmaps = dist.view(dist.shape[0], self._h, self._w).cpu().numpy()
                predicted_best_idx = dist.min(dim=1)[0].argmin()
                is_correct = predicted_best_idx == self.best_rot_idx
                msg = "Correct!" if is_correct else "Wrong!"
                self._set_prediction_text(msg)
                min_val = self.heatmaps[predicted_best_idx].argmin()
                u_min, v_min = misc.make2d(min_val, self._w)
                self._uv = [u_min, v_min, predicted_best_idx]
                self._draw_rotations(heatmap=not self._is_switch)

    def _get_next_data(self):
        """Grabs a fresh pair of source and target data points.
        """
        self._pair_idx += 1
        self.imgs, labels, center = next(self._dloader)
        self.center = center[0]
        label = labels[0]
        self.xs, self.xt = self.imgs[:, :self._num_channels, :, :], self.imgs[:, self._num_channels:, :, :]
        if self._num_channels == 4:
            self._xs_np = ml.tensor2ndarray(self.xs[:, :3], [self._color_mean * 3, self._color_std * 3])
            self._xt_np = ml.tensor2ndarray(self.xt[:, :3], [self._color_mean * 3, self._color_std * 3])
        else:
            self._xs_np = ml.tensor2ndarray(self.xs[:, :1], [self._color_mean, self._color_std], False)
            self._xt_np = ml.tensor2ndarray(self.xt[:, :1], [self._color_mean, self._color_std], False)
            self._xs_np = np.uint8(cm.viridis(self._xs_np) * 255)[..., :3]
            self._xt_np = np.uint8(cm.viridis(self._xt_np) * 255)[..., :3]
        source_idxs = label[:, 0:2]
        target_idxs = label[:, 2:4]
        rot_idx = label[:, 4]
        is_match = label[:, 5]
        self.best_rot_idx = rot_idx[0].item()
        mask = (is_match == 1) & (rot_idx == self.best_rot_idx)
        self.source_pixel_idxs = source_idxs[mask].numpy()
        self.target_pixel_idxs = target_idxs[mask].numpy()

    def _forward_network(self):
        """Forwards the current source-target pair through the network.
        """
        self.imgs = self.imgs.to(self._dev)
        with torch.no_grad():
            self.outs, self.out_t = self._net(self.imgs, *self.center)
        self.outs = self.outs[0]

    def _advance_progress_bar(self):
        """Advances the progress bar.
        """
        curr_val = self._pair_idx
        max_val = self._progress_bar.maximum()
        self._progress_bar.setValue(curr_val + (max_val - curr_val) / 100)

    def _add_border_clr(self, img, color):
        """Adds a border color to an image.
        """
        img[0 : self._h - 1, 0:10] = color  # left
        img[0:10, 0 : self._w - 1] = color  # top
        img[self._h - 11 : self._h - 1, 0 : self._w - 1] = color
        img[0 : self._h - 1, self._w - 11 : self._w - 1] = color
        return img


if __name__ == "__main__":
    def str2bool(s):
        return s.lower() in ["1", "true"]
    parser = argparse.ArgumentParser(description="Descriptor Network Visualizer")
    parser.add_argument("foldername", type=str)
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--num_desc", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--background_subtract", type=tuple, default=None)
    parser.add_argument("--augment", type=str2bool, default=False)
    args, unparsed = parser.parse_known_args()
    app = QApplication(sys.argv)
    window = Debugger(args)
    window.show()
    sys.exit(app.exec_())
