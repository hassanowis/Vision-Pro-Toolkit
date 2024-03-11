from PyQt5.QtCore import *
import functools
import sys
from os import path
import cv2
import numpy as np
import pyqtgraph as pg
import logging
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QRubberBand,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSizePolicy,
)
from PyQt5.uic import loadUiType
# from image import Image
# from imageMixer import ImageMixer
# from overlay import overlay

FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "mainwindow.ui")
)  # connects the Ui file with the Python file

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super().__init__()
        QMainWindow.__init__(self)
        self.setupUi(self)
    def import_image(self):
          """
          Browse and open an image file.

          """

          image_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.jpg *.gif *.png *.jpeg *.svg)"
          )
          if image_path:
            self.image = Image(image_path)
            self.image.load_image()
          
          
def main():
    """
    Main method to start the application.
    """
    logging.info('Starting application')
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()  # infinite Loop
    logging.info('Exiting application')


if __name__ == "__main__":
    main()