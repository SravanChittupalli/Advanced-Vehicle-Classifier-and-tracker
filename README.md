# Advanced-Vehicle-Classifier  🚗 🚛 🚴🏽 🚍
This project aims at classifying multiple classes of vehicles and in addition to that sub-classifying of trucks according to number of axels. 

# Contents
- [Why this project](#why-this-project)
- [About the Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [To-Do](#to-do)

# Why this project
- Sub-classification of trucks is a very important task as it has many uses like toll-automation and counting number and type of vehicles accurately. Specially in India all types of trucks look the same untill you are a truck connosuir but the wear and tear done by different trucks depends on the number of tyres. So each vehicle has to be charged differently.

- You can also see that at toll gates where it is completely manual the workers can misinform the authorities about the total amount earned for that day thus leading to loss for the authority owning the road. This project can reduce such frauds.

- Nowadays we find the use of RFID tags for toll automation. The problem with this is every user has to buy one but by using _ComputerVision_ no external hardware is required. Once setup you will get the data automatically 24X7. 

# About the Project
The project contains a GUI application to generalise the project. It can be used at toll gates to keep a good track of number of vehicles and with addition of some more classes the project can also be used for full toll automation. The project uses Darknet YOLOv4 and SORT tracker.
# Tech Stack
* [Python](https://www.python.org/)
* [Numpy](https://numpy.org)
* [SciPy](https://pypi.org/project/scipy/1.5.1/)
* [OpenCV](https://opencv.org/)
* [PyQt5](https://pypi.org/project/PyQt5/)


# To-Do
I've tried this project using feature extraction , Hough Circle detection , area of truck , counting each wheel individually but atlast ended up using 2 iterations of the neural network on 2 different weights :sweat_smile:. I understand that this is not an efficient method but I did not have enough resources to make a whole dataset by myself which includes cars, trucks , bikes , buses along with their wheels. If anyone has any sugestions on solving this problem efficintly them please open an issue I will be more than happy to try and implement the suggestions or else you can even try to implement it on your own. :smile:
