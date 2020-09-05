# Advanced-Vehicle-Classifier-And-Tracker  🚗 🚛 🚴🏽 🚍
This project aims at classifying multiple classes of vehicles and in addition to that sub-classifying of trucks according to number of axels. 

## Contents
- [Why this project](#why-this-project)
- [About the Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [File Structure](#file-structure)
- [How to run the application](#how-to-run-the-application)
- [Demo](#demo)
- [To-Do](#to-do)
- [References](#references)
- [License](#license)

## Why this project
- Sub-classification of trucks is a very important task as it has many uses like toll-automation and counting number and type of vehicles accurately. Specially in India all types of trucks look the same untill you are a truck connoisseur but the wear and tear done by different trucks depends on the number of tyres. So each vehicle has to be charged differently.

- You can also see that at toll gates where it is completely manual the workers can misinform the authorities about the total amount earned for that day thus leading to loss for the authority owning the road. This project can reduce such frauds.

- Nowadays we find the use of RFID tags for toll automation. The problem with this is every user has to buy one but by using _ComputerVision_ no external hardware is required. Once setup you will get the data automatically 24X7. 

## About the Project
![GUI](https://github.com/SravanChittupalli/Advanced-Vehicle-Classifier/blob/master/Code/media/Pics_Readme/GUI.png)

The project contains a GUI application to generalise the project. It can be used at toll gates to keep an accurate track of number of vehicles and with addition of some more classes and features the project can also be used for full toll automation. The project uses Darknet YOLOv4 and SORT tracker. For now the app can only classify car , bike , bus and 5 classes of trucks.

## Tech Stack
* [Python](https://www.python.org/)
* [Numpy](https://numpy.org)
* [SciPy](https://pypi.org/project/scipy/1.5.1/)
* [OpenCV](https://opencv.org/)
* [PyQt5](https://pypi.org/project/PyQt5/)


## File Structure
    ├── LICENSE
    ├── README.md                
    ├── WORKPLAN.md                       # Project plan
    ├── Code/
    │   ├── GUIApp.py                     # GUI APP
    │   ├── SortTracker.py                # SortImplementation
    │   ├── classify_track_count.py       # YOLO classifier
    |   ├── requirements.txt              # All required libraries
    │   ├── extras            
    │      |-- coco.data           
    |      |-- coco.names
    |      |-- multi_classify.cfg
    |      |-- multi_classify.data
    |      |-- multi_classify.names
    │      |-- yolov4.cfg         
    |   |-- media/Pics_Readme
    |      |-- GUI.png
    |      |-- fulldemo.gif

## How to run the application
This is the problem with darknet. I can't find a way to give the whole project as a package along with darknet. Please do the following steps extremely carefully.
  1) Clone [AlexyAB's darknet](https://github.com/SravanChittupalli/darknet) github repo.
  2) Run the python demo as given in the [README](https://github.com/AlexeyAB/darknet/blob/master/README.md). If you built and ran the demo successfully then continue to step 3
  3) Clone this repo into the `darknet` folder
  4) Next copy the 3 python files in `Code` folder into the `darknet` folder
  5) Copy the `.data` and `.names` files in `Code/extras` to `darknet/data` folder.
  6) Copy the `.cfg` files in `Code/extras` to `darknet/cfg` folder.
  7) Download the [weight](https://drive.google.com/drive/u/0/folders/1XVWolAhNTvv-ssePnYNXk0GNMrzmwN0w) files from the [drive link](https://drive.google.com/drive/u/0/folders/1XVWolAhNTvv-ssePnYNXk0GNMrzmwN0w) and save them in `darknet` folder. There will also be a `sample_videos` folder also place it anywhere you want.
  8) Open a terminal and run `pip install -r requirements.txt`. I strongly recommend the use of miniconda as is keep the system python packages seperate, thus avoiding conflict.
  9) Now you are all set to run the demo. Activate your environment and run `python GUIApp.py`.
  10) Select the sample video and choose the ROI as shown in the GIF below. Detection of axels requires specific angles so I recommend using the ROI as `(444,191), (1440,734)` to get the best results.
  
  Just to make your life easier i've added a [bash script]() that you can run to do everything from `step 4` to `step 6`

## Demo
![WORKING DEMO](https://github.com/SravanChittupalli/Advanced-Vehicle-Classifier/blob/master/Code/media/Pics_Readme/fulldemo.gif)

## To-Do
I've tried this project using feature extraction , Hough Circle detection , Finding area and length of truck , counting each wheel individually but atlast ended up using 2 iterations of the neural network on 2 different weights one which classifies vehicles and the other that detects wheels :sweat_smile:. I understand that this is not an efficient method but I did not have enough resources to make a whole dataset by myself which includes cars, trucks , bikes , buses along with their wheels. Also I could not find many videos with trck , car facing to the side so that I can detect wheels easily. If anyone has any sugestions on solving this problem efficintly them please open an issue I will be more than happy to try and implement the suggestions or else you can even try to implement it on your own. :smile:
- [x] Add script to ensure smooth running
- [ ] increase # of classes
- [ ] add number plate recognition
- [ ] increase efficiency

## References
* [AlexyAB's YOLOV4](https://github.com/AlexeyAB/darknet)
* [SORT Implementation](https://github.com/abewley/sort)
* [Point's side w.r.t a line](https://www.geeksforgeeks.org/direction-point-line-segment/)
* Thanks to pyimagesearch. There is are no topics in CV and ImageProcessing that pyimagesearch blog doesnot cover. 

## License
Details can be found in [License](LICENSE). 
