# -*- coding: utf-8 -*-



from NST import NST
import scipy.misc
from ui2 import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import *
import sys



#image

class StyleTransferWindow(QtWidgets.QMainWindow):
    content_image_path=""
    style_image_path=""
    generated_image=None
    def __init__(self):
        super(StyleTransferWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_2.clicked.connect(self.load_file1)
        self.ui.pushButton.clicked.connect(self.load_file2)
        self.ui.pushButton_4.clicked.connect(self.Transfer)
        self.ui.pushButton_3.clicked.connect(self.save_file)
        
        
    def test(self):
        print("Hello!")
        
    def load_file1(self):
       # print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'd:\\', 'Image files(*.jpg *.gif *.png)')
        self.ui.lineEdit.setText(fname)
      #  print(fname)
        scene = QGraphicsScene()
        pic = QPixmap(fname)
        scene.addItem(QGraphicsPixmapItem(pic))
        view = self.ui.graphicsView
        view.setScene(scene)
        view.setRenderHint(QPainter.Antialiasing)
        view.show()
        # self.label.setPixmap(QPixmap(fname))
        self.content_image_path=fname

    def load_file2(self):
       # print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'd:\\', 'Image files(*.jpg *.gif *.png)')
        self.ui.lineEdit_2.setText(fname)
      #  print(fname)
        scene = QGraphicsScene()
        pic = QPixmap(fname)
        scene.addItem(QGraphicsPixmapItem(pic))
        view = self.ui.graphicsView_3
        view.setScene(scene)
        view.setRenderHint(QPainter.Antialiasing)
        view.show()
        # self.label.setPixmap(QPixmap(fname))
        self.style_image_path=fname


    def Transfer(self):
        self.ui.label_2.setText("Transfer start!")
        self.generated_image=NST(self.content_image_path,self.style_image_path)
    
        #img=scipy.misc.imread(self.content_image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(2)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                #转换图像通道        
        #frame = QImage(img)
        print(1)        
        scene = QGraphicsScene()
        pic = QPixmap("output/generated_image.png")
        scene.addItem(QGraphicsPixmapItem(pic))
        view = self.ui.graphicsView_2
        view.setScene(scene)
        view.setRenderHint(QPainter.Antialiasing)
        view.show()
        self.ui.label_2.setText("Transfer complete!")
        # self.label.setPixmap(QPixmap(fname))
        


    def save_file(self):
        print("save--file")
        fname, _=QFileDialog.getSaveFileName(self, "保存图片", 'd:\\', 'Image files(*.jpg *.gif *.png)')
        self.ui.lineEdit_3.setText(fname)
        print(fname)
        save_image(fname,self.generated_image)
        #scipy.misc.imsave(fname, self.generated_image)
       # return fname
        
        
        
        

def main2():
    app = QtWidgets.QApplication(sys.argv)
    application = StyleTransferWindow()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main2()