from PyQt5 import QtWidgets, uic, QtGui
import sys
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
from PIL import Image
import matplotlib.pyplot as plt
class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()

        global originalImage, pil_image
        originalImage = None
        uic.loadUi('mainWindow.ui', self)
        self.loadImgBtn.clicked.connect(self.loadImgFunc)
        self.LogTransformBtn.clicked.connect(self.LogTransform)
        self.medianBtn.clicked.connect(self.medianProcess)
        self.GaussianBtn.clicked.connect(self.gaussiann)
        self.negativeBtn.clicked.connect(self.negativeImage)
        self.powerBtn.clicked.connect(self.powerLaw)
        self.RBtn.clicked.connect(self.RChannel)
        self.GBtn.clicked.connect(self.GChannel)
        self.BBtn.clicked.connect(self.BChannel)
        self.HistogramBtn.clicked.connect(self.histogram)
        self.AverageBtn.clicked.connect(self.averageColour)
        self.resizeBtn.clicked.connect(self.resizeProcess)
        self.rgbBTn.clicked.connect(self.convertToRGB)
        self.additionBtn.clicked.connect(self.additionImage)
        self.subtractBtn.clicked.connect(self.subtractImage)
        self.cropBtn.clicked.connect(self.extractRegionImage)
        self.equalizationBtn.clicked.connect(self.histogramEqualization)
        self.saveBtn.clicked.connect(self.saveImg)
        self.KernelSharpenBtn.clicked.connect(self.KernelSharpen)
        self.KernelAverageBtn.clicked.connect(self.KernelAverage)
        self.ContrastAndBrightnessBtn.clicked.connect(self.ContrastAndBrightness)
        self.show()


    def loadImgFunc(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '/home/', "Images (*.png *.jpeg *.jpg)")
        if len(image[0]) != 0:
            print(image)
            global imagePath
            imagePath = image[0]
            if imagePath is not None:
                pixmap = QPixmap(imagePath)
                global originalImage, pil_image
                originalImage = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
                pil_image = Image.open(imagePath)
                self.imagePathTxt.setText(imagePath)
                self.mainImage.setPixmap(self.convertcvImgToQtImg(originalImage))
                self.mainImage.setScaledContents(1)
    
    def additionImage(self):
        if originalImage is not None:
            image = QFileDialog.getOpenFileName(None, 'OpenFile', '/home/', "Images (*.png *.jpeg *.jpg)")

            if len(image[0]) != 0:
                SecondimagePath = image[0]
                Secondimage = cv2.imread(SecondimagePath, cv2.IMREAD_UNCHANGED)
                dimensions = (originalImage.shape[1],originalImage.shape[0])
                Secondimage = cv2.resize(Secondimage, dimensions, interpolation=cv2.INTER_AREA)
                weightedSum = cv2.addWeighted(originalImage, 0.5, Secondimage, 0.4, 0)
                self.processedImage.setPixmap(self.convertcvImgToQtImg(weightedSum))
                self.processedImage.setScaledContents(1)
    
    def LogTransform(self): 
        if originalImage is not None:
            # Apply log transformation method
            c = 255 / np.log(1 + np.max(originalImage))
            log_image = c * (np.log(originalImage + 1))
   
            # Specify the data type so that
            # float value will be converted to int
            log_image = np.array(log_image, dtype = np.uint8)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(log_image))
            self.processedImage.setScaledContents(1)

    def averageColour(self):
        if originalImage is not None:
            average_color_row = np.average(originalImage, axis=0)
            average_color = np.average(average_color_row, axis=0)
            d_img = np.ones((312, 312, 3), dtype=np.uint8)
            d_img[:, :] = average_color
            self.processedImage.setPixmap(self.convertcvImgToQtImg(d_img))
            self.processedImage.setScaledContents(1)

    def histogram(self):
        if originalImage is not None:
            # hist = cv2.calcHist([originalImage], [0], None, [256], [0, 256])
            img = cv2.imread(imagePath)
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])

            figure = plt.gcf()
            figure.canvas.draw()
            b = figure.axes[0].get_window_extent()
            img = np.array(figure.canvas.buffer_rgba())
            img = img[int(b.y0):int(b.y1), int(b.x0):int(b.x1), :]
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(img))
            self.processedImage.setScaledContents(1)
            
    def subtractImage(self):
        if originalImage is not None:
            image = QFileDialog.getOpenFileName(None, 'OpenFile', '/home/', "Images (*.png *.jpeg *.jpg)")
            if len(image[0]) != 0:
                SecondimagePath = image[0]
                Secondimage = cv2.imread(SecondimagePath, cv2.IMREAD_UNCHANGED)
                dimensions = (originalImage.shape[1], originalImage.shape[0])
                Secondimage = cv2.resize(Secondimage, dimensions, interpolation=cv2.INTER_AREA)
                sub = cv2.subtract(originalImage, Secondimage)
                self.processedImage.setPixmap(self.convertcvImgToQtImg(sub))
                self.processedImage.setScaledContents(1)

    def powerLaw(self):
        if originalImage is not None:
            gamma_point_four = np.array(255 * (originalImage / 255) ** self.powerInput.value(), dtype='uint8')
            self.processedImage.setPixmap(self.convertcvImgToQtImg(gamma_point_four))
            self.processedImage.setScaledContents(1)          
    def negativeImage(self):
        if originalImage is not None:
            img_neg = 255 - originalImage
            #img_neg = cv2.bitwise_not(originalImage)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(img_neg))
            self.processedImage.setScaledContents(1)
    def ContrastAndBrightness(self):
        if originalImage is not None:
            img = originalImage
            CBimg = cv2.addWeighted( img, int(self.ContrastInput.value()), img, 0, int(self.BrightnessInput.value()))
            self.processedImage.setPixmap(self.convertcvImgToQtImg(CBimg))
            self.processedImage.setScaledContents(1)
    def convertToRGB(self):
        if originalImage is not None:
            image_rgb = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(image_rgb))
            self.processedImage.setScaledContents(1)
    def BChannel(self):
        if originalImage is not None:
            image_b = originalImage
            b,g,r = cv2.split(image_b)
            imgheight = image_b.shape[0]
            imgwidth = image_b.shape[1]
            for y in range(imgheight):
                for x in range(imgwidth):
                    image_b[y][x] = (self.RInput.value(),g[y][x],r[y][x])
            self.processedImage.setPixmap(self.convertcvImgToQtImg(image_b))
            self.processedImage.setScaledContents(1)
    def GChannel(self):
        if originalImage is not None:
            image_g = originalImage
            b,g,r = cv2.split(image_g)
            imgheight = image_g.shape[0]
            imgwidth = image_g.shape[1]
            for y in range(imgheight):
                for x in range(imgwidth):
                    image_g[y][x] = (b[y][x],self.GInput.value(),r[y][x])
            self.processedImage.setPixmap(self.convertcvImgToQtImg(image_g))
            self.processedImage.setScaledContents(1)
    def RChannel(self):
        if originalImage is not None:
            image_r = originalImage
            b,g,r = cv2.split(image_r)
            imgheight = image_r.shape[0]
            imgwidth = image_r.shape[1]
            for y in range(imgheight):
                for x in range(imgwidth):
                    image_r[y][x] = (b[y][x], g[y][x],self.BInput.value())
            self.processedImage.setPixmap(self.convertcvImgToQtImg(image_r))
            self.processedImage.setScaledContents(1)
    def medianProcess(self):
        if originalImage is not None:
            median = cv2.medianBlur(originalImage, int(self.medianValBox.value()))
            self.processedImage.setPixmap(self.convertcvImgToQtImg(median))
            self.processedImage.setScaledContents(1)
    def gaussiann(self):
        if originalImage is not None:
            dst = cv2.GaussianBlur(originalImage, (int(self.ugaussianInput.value()), int(self.vgaussianInput.value())), cv2.BORDER_DEFAULT)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(dst))
            self.processedImage.setScaledContents(1)

    def extractRegionImage(self):
        if originalImage is not None:
            blank = np.zeros(originalImage.shape[:2], dtype='uint8')
            center_coordinates = (int(originalImage.shape[1]/2), int(originalImage.shape[0]/2))
            radius = 350
            mask = cv2.circle(blank, center_coordinates, radius, 255, -1)
            masked = cv2.bitwise_and(originalImage, originalImage, mask=mask)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(masked))
            self.processedImage.setScaledContents(1)

    def resizeProcess(self):
        if originalImage is not None:
            width = int(self.heightInput.value())
            height = int(self.widthInput.value())
            dim = (width, height)
            resized = cv2.resize(originalImage, dim, interpolation=cv2.INTER_AREA)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(resized))
            self.processedImage.setScaledContents(1)
    def convertcvImgToQtImg(self,cvImage):
         cvImage = QtGui.QImage(cvImage, cvImage.shape[1], cvImage.shape[0], cvImage.shape[1] * 3, QtGui.QImage.Format_BGR888)
         pix = QtGui.QPixmap(cvImage)
         return QtGui.QPixmap(pix)           

    def saveImg(self):
        if self.processedImage.pixmap() is not None:
            self.processedImage.pixmap().save('ProcessedImage/ProcessedImage.png')

    def KernelSharpen(self):
        if originalImage is not None: 
            kernel_sharpening = np.array([[-1,-1,-1], 
                                          [-1, 9,-1],
                                          [-1,-1,-1]])
            Kenhance = cv2.filter2D(originalImage, -1, kernel_sharpening)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(Kenhance))
            self.processedImage.setScaledContents(1)

    def KernelAverage(self):
        if originalImage is not None: 
            kernelaveraging = np.ones((5,5),np.float32)/25
            AKimg = cv2.filter2D(originalImage, -1, kernelaveraging)
            self.processedImage.setPixmap(self.convertcvImgToQtImg(AKimg))
            self.processedImage.setScaledContents(1)

    def histogramEqualization(self):
        if originalImage is not None:
      # segregate color streams
            b, g, r = cv2.split(originalImage)
            h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
            h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
            h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
            # calculate cdf
            cdf_b = np.cumsum(h_b)
            cdf_g = np.cumsum(h_g)
            cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
            cdf_m_b = np.ma.masked_equal(cdf_b, 0)
            cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
            cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

            cdf_m_g = np.ma.masked_equal(cdf_g, 0)
            cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
            cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')


            cdf_m_r = np.ma.masked_equal(cdf_r, 0)
            cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
            cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
            # merge the images in the three channels
            img_b = cdf_final_b[b]
            img_g = cdf_final_g[g]
            img_r = cdf_final_r[r]

            img_out = cv2.merge((img_b, img_g, img_r))
            # validation
            equ_b = cv2.equalizeHist(b)
            equ_g = cv2.equalizeHist(g)
            equ_r = cv2.equalizeHist(r)
            equ = cv2.merge((equ_b, equ_g, equ_r))
        self.processedImage.setPixmap(self.convertcvImgToQtImg(equ))
        self.processedImage.setScaledContents(1)


app = QtWidgets.QApplication(sys.argv)
window = Ui()

app.exec_()

