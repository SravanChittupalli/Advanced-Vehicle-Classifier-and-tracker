from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import csv
import time

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
from SortTracker import *
tracker = Sort()
memory = {}
counter = 0
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        if (detection[1] * 100) > 90:
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
    return img

netMain = None
metaMain = None
altNames = None

netMain1 = None
metaMain1 = None
altNames1 = None

car_cnt = 0
truck_cnt = 0
motorbike_cnt = 0
bus_cnt = 0
entry = {}
speed = 0
time_last = 0
last_count = 0
count_inc = 0
time_interval = 0
twoaxel = 0
threeaxel = 0
fouraxel = 0
fiveaxel = 0
sixaxel = 0
cur_frame = 0

def YOLO(roipt1 , roipt2 , progressObj , filename):
    global metaMain, netMain, altNames , metaMain1 , netMain1 , altNames1 , car_cnt , truck_cnt , motorbike_cnt , bus_cnt , time_last , last_count , count_inc , cur_frame , time_interval ,  memory , counter , twoaxel, threeaxel, fouraxel, fiveaxel, sixaxel
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"

    configPath1 = "./cfg/multi_classify.cfg"
    weightPath1 = "./weights/multiclassify_4000.weights"
    metaPath1 = "./data/multi_classify.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass



    if not os.path.exists(configPath1):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath1)+"`")
    if not os.path.exists(weightPath1):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath1)+"`")
    if not os.path.exists(metaPath1):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath1)+"`")
    if netMain1 is None:
        netMain1 = darknet.load_net_custom(configPath1.encode(
            "ascii"), weightPath1.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain1 is None:
        metaMain1 = darknet.load_meta(metaPath1.encode("ascii"))
    if altNames1 is None:
        try:
            with open(metaPath1) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames1 = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    
    cap = cv2.VideoCapture(filename)
    print(cap.get(cv2.CAP_PROP_FPS))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "axelCount_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    
    darknet_image1 = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    while True:
        print(cur_frame*100/tot_frames)
        progressObj.setProperty("value", cur_frame*100/tot_frames)
        prev_time = time.time()
        ret, frame_read = cap.read()
        cur_frame+=1
        if ret:
            ROI = np.copy(frame_read)
            ROI[: , : , :] = 0
            reqdRegion = np.copy(frame_read)
            reqdRegion[: , : , :] = 0
            roipi1x = roipt1[0]
            roipt1y = roipt1[1]
            roipt2x = roipt2[0]
            roipt2y = roipt2[1]
            #region = frame_read[240:790 , 400:1400]
            #ROI[240:790 , 400:1400] = region
            #cv2.rectangle(frame_read , (400 , 240) , (1400 , 790) , (173 , 50, 200) , 2)
            region = frame_read[roipt1y:roipt2y , roipi1x:roipt2x]
            ROI[roipt1y:roipt2y , roipi1x:roipt2x] = region
            cv2.rectangle(frame_read , roipt1 , roipt2 , (173 , 50, 200) , 2)
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            
            frame_resized = cv2.resize(frame_rgb,
                                    (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)
            ROI_resized = cv2.resize(ROI,
                                    (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)
            reqdRegion_resized = cv2.resize(reqdRegion,
                                    (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image , ROI_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            image = cvDrawBoxes(detections, frame_resized)
            image_clean = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
            dets = []
            full_dets=[]
            if len(detections) > 0:
            # loop over the indexes we are keeping
                for i in range (0,len(detections)):
                    if detections[i][0].decode() != 'person':
                        #print(len(detections))
                        (x, y) = (detections[i][2][0], detections[i][2][1])
                        (w, h) = (detections[i][2][2] , detections[i][2][3] )
                        dets.append([float(x-w/2), float(y-h/2), float(x+w/2), float(y+h/2), float(detections[i][1])])
                        full_dets.append([int((float(x-w/2)+float(x+w/2))/2) , int((float(y-h/2)+float(y+h/2))/2), detections[i][0].decode()])
                        #print(np.shape(dets))
                        #print(full_dets)
            
            print(full_dets)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            dets = np.asarray(dets)
            print(dets)
            print("##################################")
            tracks = tracker.update(dets)
            #print(tracks)

            boxes = []
            indexIDs = []
            c = []
            previous = memory.copy()
            memory = {}

            for track in tracks:
                # As boxes co-ordinates and indexes are appended one by one we are storing them
                # in a dictionary
                #print(track)
                for i , det_cent in enumerate(full_dets):
                    euclidean_distance = math.sqrt( (det_cent[0]-(track[0]+track[2])/2)**2 + (det_cent[1]-(track[1]+track[3])/2)**2 )
                    if euclidean_distance <= 4:
                        boxes.append([track[0], track[1], track[2], track[3], det_cent[2]])
                        indexIDs.append(int(track[4]))
                        memory[indexIDs[-1]] = boxes[-1]
                        break
            
            if len(boxes) > 0:
                i = int(0)
                for box in boxes:
                    # extract the bounding box coordinates
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))
                    cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (255 , 0 , 0), 2)

                    if indexIDs[i] in previous:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        # p0 = prevCentroid
                        # p1 = current centroid
                        # condition for counter to increment is if line between prev centroid and current centroid intersect then increment counter
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                        cv2.putText(image, str(indexIDs[i]), p0 , cv2.FONT_HERSHEY_SIMPLEX , 0.5,(0, 255, 0) , 2)
                        cv2.line(image , p0, p1, (0 , 255 , 0), 2)
                        
                        countwheel=0
                        # If diagonal intersects with line then increment counter
                        if intersect(p0, p1, (340 , 366) , (340 , 138)):
                            print('entered')
                            
                            if box[4] == 'car':
                                car_cnt+=1
                            if box[4] == 'truck':
                                print("Counting axels...........")
                                region = image_clean[y-20:h+20 , x-20:w+20]
                                reqdRegion_resized[y-20:h+20 , x-20:w+20] = region
                                reqdRegion_resized = cv2.cvtColor(reqdRegion_resized , cv2.COLOR_RGB2BGR)
                                darknet.copy_image_from_bytes(darknet_image1 , reqdRegion_resized.tobytes())
                                detections = darknet.detect_image(netMain1, metaMain1, darknet_image1, thresh=0.25)
                                reqdRegion_resized = cv2.cvtColor(reqdRegion_resized , cv2.COLOR_BGR2RGB)
                                reqdRegion_resized = cvDrawBoxes(detections, reqdRegion_resized)
                                if len(detections) > 0:
                                    for j in range (0,len(detections)):
                                        if detections[j][0].decode() == 'wheel':
                                            countwheel+=1
                                print('No of wheeles are: ' , countwheel)
                                #cv2.imshow("img" , reqdRegion_resized)
                                #cv2.waitKey(0)
                                print("Done counting..........")
                                truck_cnt+=1
                                if countwheel == 2:
                                    twoaxel+=1
                                if countwheel == 3:
                                    threeaxel+=1
                                if countwheel == 4:
                                    fouraxel+=1
                                if countwheel == 5:
                                    fiveaxel+=1
                                if countwheel == 6:
                                    sixaxel+=1
                            if box[4] == 'motorbike':
                                motorbike_cnt+=1
                            if box[4] == 'bus':
                                bus_cnt+=1
                            counter += 1
                    i+=1
            cv2.putText(image,
                    "Total Count = {}".format(counter),
                    (10 , 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "Car Count = {}".format(car_cnt),
                    (10 , 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "Truck Count = {}".format(truck_cnt),
                    (10 , 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "2-Axel Truck Count = {}".format(twoaxel),
                    (20 , 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "3-Axel Truck Count = {}".format(threeaxel),
                    (20 , 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "4-Axel Truck Count = {}".format(fouraxel),
                    (20 , 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "5-Axel Truck Count = {}".format(fiveaxel),
                    (20 , 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "6-Axel Truck Count = {}".format(sixaxel),
                    (20 , 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "Motorbike Count = {}".format(motorbike_cnt),
                    (10 , 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "Bus Count = {}".format(bus_cnt),
                    (10 , 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2) 
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
            cv2.line(image , (340 , 366) , (340 , 138) , (255 , 0 , 0) , 2 )
            #print(1/(time.time()-prev_time))
            out.write(image)
            #cv2.imshow('original' , frame_read)
            cv2.imshow('Demo', image)
            #cv2.imshow('ROI' , ROI)
            k = cv2.waitKey(3)
            if k&0xFF ==  ord('q'):
                break
        else:
            print("DONE")    
            break
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
