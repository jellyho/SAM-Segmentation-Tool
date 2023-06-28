import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPalette, QPixmap
from PyQt5.QtCore import *
from PyQt5 import uic
import cv2
from segment_anything import sam_model_registry, SamPredictor
import glob, sys, os

form_class = uic.loadUiType("ui.ui")[0]
BASE_DIR = 'E:/WorkSpace/Python/SST_Datasets'

class SST(QMainWindow, form_class):
    def __init__(self, base_dir):
        super().__init__()
        self.setWindowTitle('SAM Segmentation Tool')
        self.setupUi(self)
        self.pixmap.setBackgroundRole(QPalette.Base)
        self.pixmap.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.pixmap.setScaledContents(True)
        self.previous.clicked.connect(self.btn_prev)
        self.next.clicked.connect(self.btn_next)
        self.make.clicked.connect(self.btn_make)
        self.Retry.clicked.connect(self.btn_retry)
        self.base_dir = base_dir
        self.idx = 0
        self.layeridx = 0
        self.ready = False
        if not os.path.isdir(base_dir+'/Mask'):
            print('Create Mask Folder')
            os.mkdir(base_dir+'/Mask')
        self.img_list = glob.glob(self.base_dir + '/Img/*')
        self.filenameBox.addItems(self.img_list)
        self.filenameBox.currentIndexChanged.connect(self.select)
        self.layerBox.currentIndexChanged.connect(self.selectLayer)
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(device='cuda')
        self.predictor = SamPredictor(self.sam)
        self.load_img()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            if pos.x() >= 70 and pos.x() <= 70+941 and pos.y() >= 100 and pos.y() <= 100+521 and self.ready:
                print("clicked")
                self.predict_mask(pos)

    def load_img(self):
        image = cv2.imread(self.img_list[self.idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.predictor.set_image(image)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.pixmap.setPixmap(pixmap)
        self.layerBox.clear()
        self.status.setText("Img Decoded")
        self.ready = True

    def predict_mask(self, pos):
        print('predict')
        h, w, ch = self.image.shape
        posx, posy = pos.x() - 70, pos.y() - 100
        x, y = 941, 521
        self.status.setText(f"({posx / x * w}, {posy / y * h}) clicked....creating masks..")
        input_point = np.array([[posx / x * w, posy / y * h ]])
        input_label = np.array([1])
        masks, _, _ = self.predictor.predict(point_coords=input_point, point_labels=input_label)
        self.masks = masks > self.predictor.model.mask_threshold
        if len(self.masks) > 0:
            self.layerBox.clear()
            self.layerBox.addItems([str(i) for i in range(len(self.masks))])
            self.selectLayer(0)

    def selectLayer(self, idx):
        if idx is not None:
            self.layeridx = idx
            self.show_mask()
    def show_mask(self):
        mask = self.masks[self.layeridx]
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * self.image
        h, w, ch = mask_image.shape
        bytes_per_line = ch * w
        image = QImage(mask_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.pixmap.setPixmap(pixmap)

    def select(self, idx):
        self.idx = idx
        self.ready = False
        self.load_img()

    def btn_prev(self):
        if self.idx != 0:
            self.idx -= 1
            self.filenameBox.setCurrentIndex(self.idx)
            self.ready = False
            self.load_img()

    def btn_next(self):
        if self.idx != len(self.img_list)-1:
            self.idx += 1
            self.filenameBox.setCurrentIndex(self.idx)
            self.ready = False
            self.load_img()

    def btn_make(self):
        if self.masks is not None:
            img_name = self.img_list[self.idx].split('\\')[-1].split('.')[0]
            with open(f"{self.base_dir}/Mask/{img_name}.npy", 'wb') as f:
                np.save(f, self.masks[self.layeridx])
            self.status.setText(f"Saved - {self.base_dir}/Mask/{img_name}.npy")

    def btn_retry(self):
        h, w, ch = self.image.shape
        bytes_per_line = ch * w
        image = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.pixmap.setPixmap(pixmap)
        self.masks = None


app = QApplication(sys.argv)
sst = SST(BASE_DIR)
sst.show()
sys.exit(app.exec_())