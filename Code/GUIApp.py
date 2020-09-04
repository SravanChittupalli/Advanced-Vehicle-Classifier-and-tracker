from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from classify_track_count import *
import os


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setWindowIcon(QtGui.QIcon("logo.png"))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(100, 120, 600, 250))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 20)
        self.gridLayout.setObjectName("gridLayout")
        self.progress = QtWidgets.QProgressBar(self.centralwidget)
        self.progress.setGeometry(QtCore.QRect(300, 330, 211, 21))
        self.progress.setProperty("value", 0)
        self.progress.setObjectName("progress")
        self.label2 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label2.setFont(font)
        self.label2.setObjectName("label2")
        self.gridLayout.addWidget(self.label2, 2, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.button_roi = QtWidgets.QPushButton(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_roi.setFont(font)
        self.button_roi.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.button_roi.setObjectName("button_roi")
        self.gridLayout.addWidget(self.button_roi, 2, 2, 1, 1)
        self.button_upload = QtWidgets.QPushButton(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_upload.setFont(font)
        self.button_upload.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.button_upload.setObjectName("button_upload")
        self.gridLayout.addWidget(self.button_upload, 0, 2, 1, 1)
        self.input2 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.input2.setFont(font)
        self.input2.setObjectName("input2")
        self.gridLayout.addWidget(self.input2, 2, 1, 1, 1)
        self.input1 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.input1.setFont(font)
        self.input1.setObjectName("input1")
        self.gridLayout.addWidget(self.input1, 0, 1, 1, 1)
        self.label1 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label1.setFont(font)
        self.label1.setObjectName("label1")
        self.gridLayout.addWidget(self.label1, 0, 0, 1, 1)
        self.start = QtWidgets.QPushButton(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.start.setFont(font)
        self.start.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.start.setObjectName("start")
        self.gridLayout.addWidget(self.start, 3, 1, 1, 1)
        self.show_results = QtWidgets.QPushButton(self.centralwidget)
        self.show_results.setGeometry(QtCore.QRect(240, 450, 320, 40))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.show_results.setFont(font)
        self.show_results.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_results.setObjectName("show_results")
        self.show_results.hide()
        self.loading = QtWidgets.QLabel(self.centralwidget)
        self.loading.setGeometry(QtCore.QRect(290, 400, 320, 40))
        font.setPointSize(14)
        self.loading.setFont(font)
        self.loading.setObjectName("loading")
        self.loading.hide()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.button_upload.clicked.connect(self.upload_video)
        self.button_roi.clicked.connect(self.select_roi)
        self.start.clicked.connect(self.run_darknet)
        self.show_results.clicked.connect(self.open_file)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Vehicle Classifier"))
        self.label2.setText(_translate("MainWindow", "ROI selected: "))
        self.button_roi.setText(_translate("MainWindow", "Choose ROI"))
        self.button_upload.setText(_translate("MainWindow", "Upload video"))
        self.input2.setText(_translate("MainWindow", ""))
        self.input1.setText(_translate("MainWindow", ""))
        self.label1.setText(_translate("MainWindow", "File Location: "))
        self.start.setText(_translate("MainWindow", "START"))
        self.loading.setText(_translate("MainWindow", "This might take a while..."))
        self.show_results.setText(_translate("MainWindow", "Show Results"))

    def clear_field(self):
            self.input1.setText("")
            self.input2.setText("")
    
    def upload_video(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "Upload Video File",
                        "",
                        "All Files (*);;Video Files (*.mp4);;Video Files (*.avi)",
                        options=options)
        if fileName:
            self.input1.setText(fileName)
            self.fileName = fileName

    def select_roi(self):
        try:
            # Mouse event function
            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.points.append((x,y))
                if event == cv2.EVENT_LBUTTONUP:
                    self.points.append((x,y))
                    print(self.points)
                #if len(self.points)%2 == 2:
                    self.p1 = self.points[len(self.points)-2]
                    self.p2 = self.points[len(self.points)-1]
                    cv2.rectangle(img, self.p1, self.p2,(0,255,0), 2 )
                    self.input2.setText("("+str(self.p1[0])+","+str(self.p1[1])+")"+ ", "+"("+str(self.p2[0])+","+str(self.p2[1])+")")
                    #points.clear()
                    #print(p1 , p2)
                #cv2.imshow("frame", img)

                    
            cap = cv2.VideoCapture(self.fileName)
            _, img = cap.read()
            cv2.putText(img, "Click and drag to select ROI", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(img, "Click 'Enter' to Proceed", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.imshow("frame", img)    
            self.points = []
            cv2.setMouseCallback("frame", click_event)
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()

        except:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Error")
            msg.setWindowIcon(QtGui.QIcon("logo.png"))
            msg.setText("File type is not supported")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.buttonClicked.connect(self.clear_field)
            x = msg.exec_()
    
    def run_darknet(self):
        self.loading.show()
        self.results, self.file_loc = YOLO(self.p1 , self.p2 , self.progress , self.fileName)#str(self.input1.text()), str(self.input2.text()))
        if self.results:
            self.loading.hide()
            self.show_results.show()
            print("Results obtained!!")   

    def open_file(self):
        os.startfile(self.file_loc)

        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
