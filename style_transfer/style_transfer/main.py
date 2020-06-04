# -*- coding: utf-8 -*-


from ui1 import Ui_MainWindow
from main2 import StyleTransferWindow
from main3 import NeuralDoodleWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import *
import sys



class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #self.ui.pushButton_2.clicked.connect(self.btn_2)
        self.ui.pushButton.clicked.connect(self.btn_1)
        self.ui.pushButton_2.clicked.connect(self.btn_2)
        self.style_transfer_window = StyleTransferWindow()
        self.neural_doodle_window = NeuralDoodleWindow()
        
        
    def test(self):
        print("Hello!")
        
        
    def btn_1(self):
        self.style_transfer_window.show()
        
        
    def btn_2(self):
        self.neural_doodle_window.show()
        

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()