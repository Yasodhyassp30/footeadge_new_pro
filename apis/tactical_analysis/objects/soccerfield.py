from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cdist
import torch


class Soccerfield:
    def __init__(self):
        self.possible_frame_points = []
        self.obtained_detections = []
        self.determined_topview_points = []
        self.determined_frame_points = []
        self.list_of_homographies = []
        self.homography = []
        self.previous_homography = []
        self.smoothened = []
        self.field_detections=[]
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )
        self.base_image_points = {
            "first_18_yard":[[348,447],[348,638]],
            "second_18_yard":[[1334,446],[1334,636]],
            "half_field":[[840,66],[840,1005]],
            "first_mid_circle":[[714,412],[714,669]],
            "second_mid_circle":[[965,412],[965,669]],
            "first_18f_yard":[[131,268],[341,268],[131,814],[341,814]],
            "second_18f_yard":[[1341,268],[1550,268],[1341,814],[1550,814]],
            "first_5_yard":[[130,417],[192,417],[130,664],[192,664]],
            "second_5_yard":[[1491,417],[1551,417],[1491,664],[1551,664]],
            "first_corner":[[127,62],[127,1020]],
            "second_corner":[[154,62],[1554,1020]],
            "first_line_bottom":[[341,1010],[341,68]],
            "second_line_bottom":[[1341,1010],[1341,68]],
            "first_p_line_bottom":[[714,1010],[714,76]],
            "second_p_line_bottom":[[961,1010],[961,76]]
        }
        self.field_named = {
        "first_18_yard":{},
        "second_18_yard":{},
        "first_half_field":{},
        "second_half_field":{},
        "first_mid_circle":{},
        "second_mid_circle":{},
        "first_18f_yard":{},
        "second_18f_yard":{},
        "first_5_yard":{},
        "second_5_yard":{}
        }
        self.field_tracker = YOLO("apis/tactical_analysis/models/2000_mid.pt")
    def segmented_points(self,segmented_part,target_size,need_dilate = True):
        segmented_image = segmented_part.cpu().numpy()
        segmented_image = (segmented_image > 0).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        if need_dilate: segmented_image = cv2.erode(segmented_image,kernel,iterations=3)
        topleft = None
        topright = None
        bottomleft= None
        bottomright=None
        try: 
            contours, _ = cv2.findContours(segmented_image, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmented_image = np.zeros(segmented_image.shape, dtype=np.uint8)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(segmented_image, [largest_contour], 0, 255, 2)
            #if need_dilate: segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
            #if need_dilate: segmented_image = cv2.dilate(segmented_image,kernel,iterations=6)
            contours, _ = cv2.findContours(segmented_image, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.009 * cv2.arcLength(largest_contour, True)
            points = cv2.approxPolyDP(largest_contour, epsilon, True)  
            points = np.array(points)
            hull = cv2.convexHull(points)
            image = np.zeros(segmented_image.shape, dtype=np.uint8)
            cv2.drawContours(image, [hull], 0, 255, 2)
            contours, _ = cv2.findContours(image, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.009 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)     
            #for point in approx:
                #cv2.circle(image, tuple(point[0]), 5, 255, -1)
                #cv2.putText(image, f'({point[0][0]}, {point[0][1]})', tuple(point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #cv2.imshow("image",image)
            #cv2.waitKey(0)
            
            if len(approx) >= 4:
                sorted_points = sorted(approx, key=lambda x: (x[0][1], x[0][0]))
                if (sorted_points[0][0][0]<sorted_points[1][0][0]):
                    topleft = sorted_points[0].tolist()[0]
                    topright =sorted_points[1].tolist()[0]
                else:
                    topleft = sorted_points[1].tolist()[0]
                    topright =sorted_points[0].tolist()[0]

                if (sorted_points[-1][0][0]<sorted_points[-2][0][0]):
                    bottomleft = sorted_points[-1].tolist()[0]
                    bottomright =sorted_points[-2].tolist()[0]
                else:
                    bottomleft = sorted_points[-2].tolist()[0]
                    bottomright =sorted_points[-1].tolist()[0]
            if topleft is None or bottomleft is None or bottomright is None or topright is None :
                return []
            else:
                topleft = np.array(topleft)
                topright = np.array(topright)
                bottomleft = np.array(bottomleft)
                bottomright = np.array(bottomright)
                scale_y = target_size[1] / segmented_image.shape[1]
                scale_x = target_size[0] / segmented_image.shape[0]
                topleft = (topleft * [scale_y, scale_x]).astype(int)
                topright = (topright * [scale_y, scale_x]).astype(int)
                bottomleft = (bottomleft * [scale_y, scale_x]).astype(int)
                bottomright = (bottomright * [scale_y, scale_x]).astype(int)
                return [topleft.tolist(),topright.tolist(),bottomleft.tolist(),bottomright.tolist()]
        except Exception as e:
            print(e)
            return []
    def segmented_points_for_fields(self,segmented_part,target_size,part,need_dilate = True):
        segmented_image = segmented_part.cpu().numpy()
        segmented_image = (segmented_image > 0).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8) 
        topleft = None
        topright = None
        bottomleft= None
        bottomright=None
        try:
            if need_dilate: segmented_image = cv2.erode(segmented_image,kernel,iterations=3)
            contours, _ = cv2.findContours(segmented_image, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmented_image = np.zeros(segmented_image.shape, dtype=np.uint8)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(segmented_image, [largest_contour], 0, 255, 2)
            #if need_dilate: segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
            if need_dilate: segmented_image = cv2.dilate(segmented_image,kernel,iterations=6)
            contours, _ = cv2.findContours(segmented_image, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            points = np.array(largest_contour)
            hull = cv2.convexHull(points)
            image = np.zeros(segmented_image.shape, dtype=np.uint8)

            cv2.drawContours(image, [hull], 0, 255, 2)

            contours, _ = cv2.findContours(image, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.009 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)   

            approx_np = approx.reshape(-1, 2)
            max_y = np.max(approx_np[:, 1])
            min_y = np.min(approx_np[:, 1])

            max_gap_y = max_y - min_y
            dividing_line = min_y + max_gap_y / 2
            top_points = approx_np[approx_np[:, 1] <= dividing_line]
            bottom_points = approx_np[approx_np[:, 1] > dividing_line]  
            top_points = top_points[np.argsort(top_points[:, 0])]
            bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
            #for point in approx:
                #cv2.circle(image, tuple(point[0]), 5, 255, -1)
                #cv2.putText(image, f'({point[0][0]}, {point[0][1]})', tuple(point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #cv2.imshow("image",segmented_image)
            #cv2.waitKey(0)
            if len(approx) >= 4:
                sorted_points = sorted(approx, key=lambda x: (x[0][1], x[0][0]))
                if (sorted_points[0][0][0]<sorted_points[1][0][0]):
                    topleft = sorted_points[0].tolist()[0]
                    topright =sorted_points[1].tolist()[0]
                else:
                    topleft = sorted_points[1].tolist()[0]
                    topright =sorted_points[0].tolist()[0]

                if (sorted_points[-1][0][0]<sorted_points[-2][0][0]):
                    bottomleft = sorted_points[-1].tolist()[0]
                    bottomright =sorted_points[-2].tolist()[0]
                else:
                    bottomleft = sorted_points[-2].tolist()[0]
                    bottomright =sorted_points[-1].tolist()[0]
            if topleft is None or bottomleft is None or bottomright is None or topright is None :
                return []
            else:
                topleft = np.array(topleft)
                topright = np.array(topright)
                bottomleft = np.array(bottomleft)
                bottomright = np.array(bottomright)
                scale_y = target_size[1] / segmented_image.shape[1]
                scale_x = target_size[0] / segmented_image.shape[0]
                topleft = (topleft * [scale_y, scale_x]).astype(int)
                topright = (topright * [scale_y, scale_x]).astype(int)
                bottomleft = (bottomleft * [scale_y, scale_x]).astype(int)
                bottomright = (bottomright * [scale_y, scale_x]).astype(int)
                if part ==1:
                    if len(bottom_points)==4:
                        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
                        bottom_points = np.array(bottom_points)
                        bottom_point_1 = (bottom_points[1]*[scale_y, scale_x]).astype(int)
                        bottom_point_2 = (bottom_points[2]*[scale_y, scale_x]).astype(int)
                        self.field_named["first_half_field"]["bottom_grad_points"] = [bottom_point_1.tolist(),bottom_point_2.tolist()]
                    else:
                        self.field_named["first_half_field"]["bottom_grad_points"] =[bottomleft.tolist(),bottomright.tolist()]
                else:
                    if len(bottom_points)==4:
                        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
                        bottom_points = np.array(bottom_points)
                        bottom_point_1 = (bottom_points[1]*[scale_y, scale_x]).astype(int)
                        bottom_point_2 = (bottom_points[2]*[scale_y, scale_x]).astype(int)
                        self.field_named["second_half_field"]["bottom_grad_points"] = [bottom_point_1.tolist(),bottom_point_2.tolist()]
                    else:
                        self.field_named["second_half_field"]["bottom_grad_points"] =[bottomleft.tolist(),bottomright.tolist()]
                return [topleft.tolist(),topright.tolist(),bottomleft.tolist(),bottomright.tolist()]
        except Exception as e:
            print(e)
            return []
    def obtain_detections(self,frame):
        self.obtained_detections = []
        #{1: '18Yard Circle', 0: '18Yard', 2: '5Yard', 3: 'First Half Central Circle', 4: 'First Half Field', 5: 'Second Half Central Circle', 6: 'Second Half Field'}
        field_results = self.field_tracker.predict(frame)[0]
        self.field_detections =sv.Detections.from_ultralytics(field_results)
        for i in range(len(self.field_detections.class_id)):
           single_object= {}
           single_object["index"]=i
           single_object["id"] = self.field_detections.class_id[i]
           if self.field_detections.class_id[i] == 5:
            single_object["id"] = 3
           if self.field_detections.class_id[i] == 6:
            single_object["id"] = 4
           single_object["box"] = self.field_detections.xyxy[i].astype(int).tolist() 
           single_object["confidence"] = self.field_detections.confidence[i]
           single_object['segment_re'] = field_results.masks.data[i]
           single_object['segment'] = []
           self.obtained_detections.append(single_object)
    def organize_detections(self):
        original = self.field_named
        try:
            self.field_named = {
                "first_18_yard":{},
                "second_18_yard":{},
                "first_half_field":{},
                "second_half_field":{},
                "first_mid_circle":{},
                "second_mid_circle":{},
                "first_18f_yard":{},
                "second_18f_yard":{},
                "first_5_yard":{},
                "second_5_yard":{}
                }
            for detection in self.obtained_detections:
                if(detection["id"]==1):
                    if self.field_named["first_18_yard"]=={}:
                        self.field_named["first_18_yard"] = detection
                    elif self.field_named["second_18_yard"] =={}:
                        if self.field_named['first_18_yard']["box"][0] < detection["box"][0]:
                            self.field_named['second_18_yard'] = detection
                        else:
                            self.field_named['first_18_yard'],self.field_named['second_18_yard'] = detection,self.field_named['first_18_yard']
                if(detection["id"]==0):
                    if self.field_named["first_18f_yard"]=={}:
                        self.field_named["first_18f_yard"] = detection
                    elif self.field_named["second_18f_yard"] =={}:
                        if self.field_named['first_18f_yard']["box"][0] < detection["box"][0]:
                            self.field_named['second_18f_yard'] = detection
                        else:
                            self.field_named['first_18f_yard'],self.field_named['second_18f_yard'] = detection,self.field_named['first_18f_yard']      
                if(detection["id"]==2):
                    if self.field_named["first_5_yard"]=={}:
                        self.field_named["first_5_yard"] = detection
                    elif self.field_named["second_5_yard"] =={}:
                        if self.field_named['first_5_yard']["box"][0] < detection["box"][0]:
                            self.field_named['second_5_yard'] = detection
                        else:
                            self.field_named['first_5_yard'],self.field_named['second_5_yard'] = detection,self.field_named['first_5_yard']
                if(detection["id"]==3 or detection["id"]==5 ):
                    if self.field_named["first_mid_circle"]=={} :
                        self.field_named["first_mid_circle"] = detection
                    elif self.field_named["second_mid_circle"] =={} and abs(self.field_named["first_mid_circle"]["box"][0] - detection["box"][0]) >20:
                        if self.field_named['first_mid_circle']["box"][2] < detection["box"][2]:
                            self.field_named['second_mid_circle'] = detection
                        else:
                            self.field_named['first_mid_circle'],self.field_named['second_mid_circle'] = detection,self.field_named['first_mid_circle']
                if(detection["id"]==4 or detection["id"]==6 ):
                    if self.field_named["first_half_field"]=={}:
                        self.field_named["first_half_field"] = detection
                    elif self.field_named["second_half_field"] =={} and abs(self.field_named["first_half_field"]["box"][0] - detection["box"][0]) >20:
                        if self.field_named['first_half_field']["box"][0] < detection["box"][0]:
                            self.field_named['second_half_field'] = detection
                        else:
                            self.field_named['first_half_field'],self.field_named['second_half_field'] = detection,self.field_named['first_half_field']

            if self.field_named["first_half_field"]!={}  and self.field_named["second_half_field"]!= {}:
                if self.field_named["second_18_yard"]=={} and self.field_named["first_18_yard"]!={}:
                    if self.field_named["first_18_yard"]["box"][2] > self.field_named["first_half_field"]["box"][2]:
                        self.field_named["second_18_yard"],self.field_named["first_18_yard"]=self.field_named["first_18_yard"],self.field_named["second_18_yard"]
                if self.field_named["second_mid_circle"]=={} and self.field_named["first_mid_circle"]!={}:
                    if self.field_named["first_mid_circle"]["box"][2] > self.field_named["first_half_field"]["box"][2]:
                        self.field_named["second_mid_circle"],self.field_named["first_mid_circle"]= self.field_named["first_mid_circle"],self.field_named["second_mid_circle"]
                if self.field_named["second_18f_yard"]=={} and self.field_named["first_18f_yard"]!={}:
                    if self.field_named["first_18f_yard"]["box"][2] > self.field_named["first_half_field"]["box"][2]:
                        self.field_named["second_18f_yard"],self.field_named["first_18f_yard"]=self.field_named["first_18f_yard"],self.field_named["second_18f_yard"]
                if self.field_named["second_5_yard"]=={} and self.field_named["first_5_yard"]!={}:
                    if self.field_named["first_5_yard"]["box"][2] > self.field_named["first_half_field"]["box"][2]:
                        self.field_named["second_5_yard"],self.field_named["first_5_yard"]=self.field_named["first_5_yard"],self.field_named["second_5_yard"]
            elif self.field_named["first_mid_circle"]!={}  and self.field_named["second_mid_circle"]!= {}:
                if self.field_named["second_18_yard"]=={} and self.field_named["first_18_yard"]!={}:
                    if self.field_named["first_18_yard"]["box"][2] > self.field_named["first_mid_circle"]["box"][2]:
                        self.field_named["second_18_yard"],self.field_named["first_18_yard"]=self.field_named["first_18_yard"],self.field_named["second_18_yard"]
                if self.field_named["second_18f_yard"]=={} and self.field_named["first_18f_yard"]!={}:
                    if self.field_named["first_18f_yard"]["box"][2] > self.field_named["first_mid_circle"]["box"][2]:
                        self.field_named["second_18f_yard"],self.field_named["first_18f_yard"]=self.field_named["first_18f_yard"],self.field_named["second_18f_yard"]
                if self.field_named["second_5_yard"]=={} and self.field_named["first_5_yard"]!={}:
                    if self.field_named["first_5_yard"]["box"][2] > self.field_named["first_mid_circle"]["box"][2]:
                        self.field_named["second_5_yard"],self.field_named["first_5_yard"]=self.field_named["first_5_yard"],self.field_named["second_5_yard"]

            if self.field_named["first_5_yard"]!={} and self.field_named["first_18_yard"]!={} and (self.field_named["first_mid_circle"]=={} or self.field_named["second_mid_circle"]=={}):
                    if self.field_named["first_5_yard"]["box"][2] > self.field_named["first_18_yard"]["box"][2]:
                        self.field_named["second_18f_yard"] = self.field_named["first_18f_yard"]
                        self.field_named["first_18f_yard"] = {}
                        self.field_named["second_5_yard"] = self.field_named["first_5_yard"]
                        self.field_named["first_5_yard"] = {}
                        self.field_named["second_half_field"] = self.field_named["first_half_field"]
                        self.field_named["first_half_field"] = {}
                        self.field_named["second_mid_circle"] = self.field_named["first_mid_circle"]
                        self.field_named["first_mid_circle"] = {}
                        self.field_named["second_18_yard"] = self.field_named["first_18_yard"]
                        self.field_named["first_18_yard"] = {}
            elif self.field_named["first_18f_yard"]!={} and (self.field_named["first_mid_circle"]=={} or self.field_named["second_mid_circle"]=={}):
                    if self.field_named["first_18f_yard"]["box"][0] > self.field_named["first_18_yard"]["box"][2] :
                        self.field_named["second_18f_yard"] = self.field_named["first_18f_yard"]
                        self.field_named["first_18f_yard"] = {}
                        self.field_named["second_5_yard"] = self.field_named["first_5_yard"]
                        self.field_named["first_5_yard"] = {}
                        self.field_named["second_half_field"] = self.field_named["first_half_field"]
                        self.field_named["first_half_field"] = {}
                        self.field_named["second_mid_circle"] = self.field_named["first_mid_circle"]
                        self.field_named["first_mid_circle"] = {}
                        self.field_named["second_18_yard"] = self.field_named["first_18_yard"]
                        self.field_named["first_18_yard"] = {}
            if self.field_named["first_mid_circle"]!={} and self.field_named["second_mid_circle"] !={}:
                first_mid_width = self.field_named["first_mid_circle"]["box"][2] - self.field_named["first_mid_circle"]["box"][0]
                second_mid_width = self.field_named["second_mid_circle"]["box"][2] - self.field_named["second_mid_circle"]["box"][0]
                if first_mid_width>second_mid_width:
                    if second_mid_width/first_mid_width < 0.6:
                        self.field_named["second_mid_circle"] ={}
                else:
                    if first_mid_width/second_mid_width < 0.6:
                        self.field_named["first_mid_circle"] ={}
        except:
            self.field_named = original
    def restructured_segment(self,frame):
        
        for detection in self.field_named:
            if self.field_named[detection]!={}:
                if detection in ["first_half_field","second_half_field","first_18f_yard","second_18f_yard"]:
                    image_data = self.field_named[detection]['segment_re']
                    if detection =="first_half_field":
                        for detection2 in self.field_named:
                            if self.field_named[detection2] !={} and  detection2 != detection and "first" in detection2:
                                image_data = torch.logical_or(image_data.bool(), self.field_named[detection2]['segment_re'].bool())
                    elif detection =="second_half_field":
                        for detection2 in self.field_named:
                            if self.field_named[detection2] !={} and detection2 != detection and "second" in detection2:
                                image_data = torch.logical_or(image_data.bool(), self.field_named[detection2]['segment_re'].bool())
                    elif detection =="first_18f_yard":
                        for detection2 in self.field_named:
                            if self.field_named[detection2] !={} and  detection2 != detection and detection2 == "first_5_yard":
                                image_data = torch.logical_or(image_data.bool(), self.field_named[detection2]['segment_re'].bool())
                    elif detection =="second_18f_yard":
                        for detection2 in self.field_named:
                            if self.field_named[detection2] !={} and  detection2 != detection and detection2 == "second_5_yard":
                                image_data = torch.logical_or(image_data.bool(), self.field_named[detection2]['segment_re'].bool())
                    if 'field' in detection:
                        if 'first' in detection:
                            self.field_named[detection]["segment"] = self.segmented_points_for_fields(segmented_part=image_data,target_size=frame.shape,part=1,need_dilate=True)
                        else:
                            self.field_named[detection]["segment"] = self.segmented_points_for_fields(segmented_part=image_data,target_size=frame.shape,part=2,need_dilate=True)
                    else:
                        self.field_named[detection]["segment"] = self.segmented_points(segmented_part=image_data,target_size=frame.shape,need_dilate=True)
                else:
                    if self.field_named[detection]["id"] in [1,2]:
                        self.field_named[detection]["segment"] = self.segmented_points(segmented_part=self.field_named[detection]["segment_re"],target_size=frame.shape,need_dilate=False)
    def frame_preprocess(self,frame):
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)  
        mask_green = cv2.inRange(hsv, (30, 50, 50), (90, 255, 255)) 
        kernel = np.ones((7, 7), np.uint8)
        closed_image = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        closed_image=cv2.erode(closed_image,kernel,iterations=3)
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [approx], 0, 255, thickness=cv2.FILLED)
        result_image = cv2.bitwise_and(frame, frame, mask=mask)
        return result_image  
    def determine_points(self):
        self.determined_frame_points = []
        self.determined_topview_points = []
        for key in self.field_named:

            if self.field_named[key] != {}:
                if key == "second_18_yard" and self.field_named[key]['segment'] !=[]:
                    self.determined_frame_points.append(self.field_named[key]['segment'][0])
                    self.determined_frame_points.append(self.field_named[key]['segment'][3])
                    self.determined_topview_points.append(self.base_image_points[key][0])
                    self.determined_topview_points.append(self.base_image_points[key][1])
                if key == "first_18_yard" and self.field_named[key]['segment'] !=[]:
                    self.determined_frame_points.append(self.field_named[key]['segment'][0])
                    self.determined_frame_points.append(self.field_named[key]['segment'][2])
                    self.determined_topview_points.append(self.base_image_points[key][0])
                    self.determined_topview_points.append(self.base_image_points[key][1])
                if key == "first_mid_circle":
                    self.determined_frame_points.append([self.field_named[key]['box'][0],self.field_named[key]['box'][1]])
                    self.determined_frame_points.append([self.field_named[key]['box'][0],self.field_named[key]['box'][3]])
                    self.determined_topview_points.append(self.base_image_points[key][0])
                    self.determined_topview_points.append(self.base_image_points[key][1])
                if key == "second_mid_circle":
                    self.determined_frame_points.append([self.field_named[key]['box'][2],self.field_named[key]['box'][3]])
                    self.determined_frame_points.append([self.field_named[key]['box'][2],self.field_named[key]['box'][1]])
                    self.determined_topview_points.append(self.base_image_points[key][0])
                    self.determined_topview_points.append(self.base_image_points[key][1])
                if key == "first_half_field":
                    if self.field_named["second_half_field"]!={} :
                        box_first_half = self.field_named["first_half_field"]['box']
                        box_second_half = self.field_named["second_half_field"]['box']
                        area_first_half = (box_first_half[2] - box_first_half[0]) * (box_first_half[3] - box_first_half[1])
                        area_second_half = (box_second_half[2] - box_second_half[0]) * (box_second_half[3] - box_second_half[1])
                        if area_first_half >= area_second_half:
                            if self.field_named["first_mid_circle"]!={}:
                                self.determined_frame_points.append(self.field_named[key]['segment'][1])
                                self.determined_frame_points.append(self.field_named[key]['segment'][3])
                                self.determined_topview_points.append(self.base_image_points["half_field"][0])
                                self.determined_topview_points.append(self.base_image_points["half_field"][1])
                            else:
                                self.determined_frame_points.append(self.field_named[key]['segment'][1])
                                self.determined_frame_points.append(self.field_named[key]['segment'][3])
                                self.determined_topview_points.append(self.base_image_points["first_p_line_bottom"][1])
                                self.determined_topview_points.append(self.base_image_points["first_p_line_bottom"][0])         
                        else:
                            if self.field_named["second_mid_circle"]!={}:
                                self.determined_frame_points.append(self.field_named["second_half_field"]['segment'][0])
                                self.determined_frame_points.append(self.field_named["second_half_field"]['segment'][2])
                                self.determined_topview_points.append(self.base_image_points["half_field"][0])
                                self.determined_topview_points.append(self.base_image_points["half_field"][1])
                            else:
                                self.determined_frame_points.append(self.field_named["second_half_field"]['segment'][0])
                                self.determined_frame_points.append(self.field_named["second_half_field"]['segment'][2])
                                self.determined_topview_points.append(self.base_image_points["second_p_line_bottom"][1])
                                self.determined_topview_points.append(self.base_image_points["second_p_line_bottom"][0])

                if key == "second_half_field" and self.field_named["first_half_field"]=={}:
                    if self.field_named["second_mid_circle"]!={}:
                        self.determined_frame_points.append(self.field_named["second_half_field"]['segment'][0])
                        self.determined_frame_points.append(self.field_named["second_half_field"]['segment'][2])
                        self.determined_topview_points.append(self.base_image_points["half_field"][0])
                        self.determined_topview_points.append(self.base_image_points["half_field"][1])
                    else:
                        self.determined_frame_points.append(self.field_named["second_half_field"]['segment'][0])
                        self.determined_frame_points.append(self.field_named["second_half_field"]['segment'][2])
                        self.determined_topview_points.append(self.base_image_points["second_p_line_bottom"][1])
                        self.determined_topview_points.append(self.base_image_points["second_p_line_bottom"][0])
                if (self.field_named["first_mid_circle"]=={} or self.field_named["second_mid_circle"]=={}) and (self.field_named["first_5_yard"]!={} or self.field_named["second_5_yard"]!={}  ) :
                    if key == "second_18f_yard" and self.field_named[key]['segment'] !=[] :          
                            field_frame_points_to_append = [self.field_named[key]['segment'][0],self.field_named[key]['segment'][1],self.field_named[key]['segment'][2],self.field_named[key]['segment'][3]]
                            field_base_points_to_append = [self.base_image_points[key][0],self.base_image_points[key][1],self.base_image_points[key][2],self.base_image_points[key][3]]
                            self.determined_frame_points += field_frame_points_to_append
                            self.determined_topview_points +=field_base_points_to_append
                    if key == "first_18f_yard" and self.field_named[key]['segment'] !=[] :
                            field_frame_points_to_append = [self.field_named[key]['segment'][0],self.field_named[key]['segment'][1],self.field_named[key]['segment'][2],self.field_named[key]['segment'][3]]
                            field_base_points_to_append = [self.base_image_points[key][0],self.base_image_points[key][1],self.base_image_points[key][2],self.base_image_points[key][3]]
                            self.determined_frame_points += field_frame_points_to_append
                            self.determined_topview_points +=field_base_points_to_append
                if key == "second_5_yard" and self.field_named[key]['segment'] !=[] :          
                        field_frame_points_to_append = [self.field_named[key]['segment'][0],self.field_named[key]['segment'][1],self.field_named[key]['segment'][2],self.field_named[key]['segment'][3]]
                        field_base_points_to_append = [self.base_image_points[key][0],self.base_image_points[key][1],self.base_image_points[key][2],self.base_image_points[key][3]]
                        self.determined_frame_points += field_frame_points_to_append
                        self.determined_topview_points +=field_base_points_to_append
                if key == "first_5_yard" and self.field_named[key]['segment'] !=[] :
                        field_frame_points_to_append = [self.field_named[key]['segment'][0],self.field_named[key]['segment'][1],self.field_named[key]['segment'][2],self.field_named[key]['segment'][3]]
                        field_base_points_to_append = [self.base_image_points[key][0],self.base_image_points[key][1],self.base_image_points[key][2],self.base_image_points[key][3]]
                        self.determined_frame_points += field_frame_points_to_append
                        self.determined_topview_points +=field_base_points_to_append  
        if self.field_named["first_5_yard"] != {} and self.field_named["first_half_field"]!={} and self.field_named["second_18_yard"]=={} and self.field_named["second_mid_circle"] != {}:
            field_frame_points_to_append = [[self.field_named['first_half_field']['segment'][0][0],self.field_named['first_half_field']['box'][1]]]
            field_base_points_to_append = [self.base_image_points["first_corner"][0]]
            self.determined_frame_points += field_frame_points_to_append
            self.determined_topview_points +=field_base_points_to_append

        if self.field_named["second_5_yard"] != {} and self.field_named["second_half_field"]!={} and self.field_named["first_18_yard"]=={} and self.field_named["first_mid_circle"] != {}:
            field_frame_points_to_append = [[self.field_named['second_half_field']['segment'][1][0],self.field_named['second_half_field']['box'][1]]]
            field_base_points_to_append = [self.base_image_points["second_corner"][0]]
            self.determined_frame_points += field_frame_points_to_append
            self.determined_topview_points +=field_base_points_to_append            
    def estimate_homography(self):
        try:
            self.homography,mask= cv2.findHomography(np.array(self.determined_topview_points),np.array(self.determined_frame_points),cv2.RANSAC,300.0)
            if len(self.list_of_homographies)==12:
                if abs(self.homography[0][2] - self.previous_homography[0][2]) <1500:
                    self.list_of_homographies.pop(0)
                    self.list_of_homographies.append(self.homography)
                mean_matrix = np.mean(self.list_of_homographies, axis=0)
                self.homography =mean_matrix
            else:
                self.list_of_homographies.append(self.homography)
            self.previous_homography = self.homography
        except:
            self.homography = self.previous_homography
            print("not enough points")
    def find_intersection(self,points):
        x1,y1,x2,y2,x3,y3,x4,y4 = points
        line1_pts = np.array([x1, y1, x2, y2], dtype=np.float32).reshape((2, 2))
        line2_pts = np.array([x3, y3, x4, y4], dtype=np.float32).reshape((2, 2))
        det = np.linalg.det([line1_pts[1] - line1_pts[0], line2_pts[1] - line2_pts[0]])
        if det == 0:
            return None
        else:
            lambda_val = np.linalg.det([line2_pts[0] - line1_pts[0], line2_pts[1] - line1_pts[0]]) / det
            intersection_point = line1_pts[0] + lambda_val * (line1_pts[1] - line1_pts[0])
            intersection_point = tuple(map(int, intersection_point))
            return intersection_point    
    def first_find_intersection(self):
        if self.field_named["first_18f_yard"]!={} and self.field_named["first_half_field"]!={} and len(self.field_named["first_18f_yard"]['segment'])==4 and len(self.field_named["first_half_field"]['segment'])==4  :
            first_bottom = self.find_intersection([
                self.field_named["first_18f_yard"]['segment'][1][0],
                self.field_named["first_18f_yard"]['segment'][1][1],
                self.field_named["first_18f_yard"]['segment'][3][0],
                self.field_named["first_18f_yard"]['segment'][3][1],
                self.field_named["first_half_field"]['bottom_grad_points'][0][0],
                self.field_named["first_half_field"]['bottom_grad_points'][0][1],
                self.field_named["first_half_field"]['bottom_grad_points'][1][0],
                self.field_named["first_half_field"]['bottom_grad_points'][1][1]
            ])
            if first_bottom!=None:
                    self.determined_topview_points.append(self.base_image_points["first_line_bottom"][0])
                    self.determined_frame_points.append([first_bottom[0],first_bottom[1]])
            second_bottom = self.find_intersection([
                self.field_named["first_18f_yard"]['segment'][1][0],
                self.field_named["first_18f_yard"]['segment'][1][1],
                self.field_named["first_18f_yard"]['segment'][3][0],
                self.field_named["first_18f_yard"]['segment'][3][1],
                self.field_named["first_half_field"]['segment'][0][0],
                self.field_named["first_half_field"]['segment'][0][1],
                self.field_named["first_half_field"]['segment'][1][0],
                self.field_named["first_half_field"]['segment'][1][1]
            ])
            if second_bottom!=None:
                    self.determined_topview_points.append(self.base_image_points["first_line_bottom"][1])
                    self.determined_frame_points.append([second_bottom[0],second_bottom[1]])   
    def second_find_intersection(self):
        if self.field_named["second_18f_yard"]!={} and self.field_named["second_half_field"]!={} and len(self.field_named["second_18f_yard"]['segment'])==4 and len(self.field_named["second_half_field"]['segment'])==4   :
            first_bottom = self.find_intersection([
                self.field_named["second_18f_yard"]['segment'][0][0],
                self.field_named["second_18f_yard"]['segment'][0][1],
                self.field_named["second_18f_yard"]['segment'][2][0],
                self.field_named["second_18f_yard"]['segment'][2][1],
                self.field_named["second_half_field"]['bottom_grad_points'][0][0],
                self.field_named["second_half_field"]['bottom_grad_points'][0][1],
                self.field_named["second_half_field"]['bottom_grad_points'][1][0],
                self.field_named["second_half_field"]['bottom_grad_points'][1][1]
            ])
            if first_bottom!=None:
                    self.determined_topview_points.append(self.base_image_points["second_line_bottom"][0])
                    self.determined_frame_points.append([first_bottom[0],first_bottom[1]])
            second_bottom = self.find_intersection([
                self.field_named["second_18f_yard"]['segment'][0][0],
                self.field_named["second_18f_yard"]['segment'][0][1],
                self.field_named["second_18f_yard"]['segment'][2][0],
                self.field_named["second_18f_yard"]['segment'][2][1],
                self.field_named["second_half_field"]['segment'][0][0],
                self.field_named["second_half_field"]['segment'][0][1],
                self.field_named["second_half_field"]['segment'][1][0],
                self.field_named["second_half_field"]['segment'][1][1]
            ])
            if second_bottom!=None:
                    self.determined_topview_points.append(self.base_image_points["second_line_bottom"][1])
                    self.determined_frame_points.append([second_bottom[0],second_bottom[1]]) 
    def calculate_parallel_first_line_points(self):
        if self.field_named["first_half_field"]!={} and self.field_named["first_mid_circle"]!={}:
            point1 = tuple(self.field_named["first_half_field"]['segment'][1])
            point2 = tuple(self.field_named["first_half_field"]['segment'][3])
            distance_point1 = tuple((self.field_named["first_mid_circle"]['box'][0],self.field_named["first_mid_circle"]['box'][1]))
            distance_point2 = tuple((self.field_named["first_mid_circle"]['box'][2],self.field_named["first_mid_circle"]['box'][1]))
            distance = np.sqrt((distance_point2[0] - distance_point1[0])**2 + (distance_point2[1] - distance_point1[1])**2)
            parallel_point1 = (
                int(point1[0] - distance ),
                int(point1[1] )
            )
            parallel_point2 = (
                int(point2[0] - distance ),
                int(point2[1] )
            )
        return parallel_point1, parallel_point2
    def calculate_parallel_second_line_points(self):
        if self.field_named["second_half_field"]!={} and self.field_named["second_mid_circle"]!={}:
            point1 = tuple(self.field_named["second_half_field"]['segment'][0])
            point2 = tuple(self.field_named["second_half_field"]['segment'][2])
            distance_point1 = tuple((self.field_named["second_mid_circle"]['box'][0],self.field_named["second_mid_circle"]['box'][3]))
            distance_point2 = tuple((self.field_named["second_mid_circle"]['box'][2],self.field_named["second_mid_circle"]['box'][3]))
            distance = np.sqrt((distance_point2[0] - distance_point1[0])**2 + (distance_point2[1] - distance_point1[1])**2)
            parallel_point1 = (
                int(point1[0] + distance ),
                int(point1[1] )
            )
            parallel_point2 = (
                int(point2[0] + distance ),
                int(point2[1] )
            )
        return parallel_point1, parallel_point2  
    def first_parallel(self):
        
        if self.field_named["first_mid_circle"]!={} and self.field_named["first_half_field"]!={} and  len(self.field_named["first_half_field"]['segment'])==4  :
            p1,p2 = self.calculate_parallel_first_line_points()
            first_bottom = self.find_intersection([
                p2[0],
                p2[1],
                p1[0],
                p1[1],
                self.field_named["first_half_field"]['segment'][2][0],
                self.field_named["first_half_field"]['segment'][2][1],
                self.field_named["first_half_field"]['segment'][3][0],
                self.field_named["first_half_field"]['segment'][3][1]
            ])
            if first_bottom!=None:
                    self.determined_topview_points.append(self.base_image_points["first_p_line_bottom"][0])
                    self.determined_frame_points.append([first_bottom[0],first_bottom[1]])    
        if self.field_named["first_mid_circle"]!={} and self.field_named["first_half_field"]!={} and  len(self.field_named["first_half_field"]['segment'])==4  :
            p1,p2 = self.calculate_parallel_first_line_points()
            first_bottom = self.find_intersection([
                p2[0],
                p2[1],
                p1[0],
                p1[1],
                self.field_named["first_half_field"]['segment'][0][0],
                self.field_named["first_half_field"]['segment'][0][1],
                self.field_named["first_half_field"]['segment'][1][0],
                self.field_named["first_half_field"]['segment'][1][1]
            ])
            if first_bottom!=None:
                    self.determined_topview_points.append(self.base_image_points["first_p_line_bottom"][1])
                    self.determined_frame_points.append([first_bottom[0],first_bottom[1]])   
    def second_parallel(self):
        
        if self.field_named["second_mid_circle"]!={} and self.field_named["second_half_field"]!={} and  len(self.field_named["second_half_field"]['segment'])==4  :
            p1,p2 = self.calculate_parallel_second_line_points()
            first_bottom = self.find_intersection([
                p2[0],
                p2[1],
                p1[0],
                p1[1],
                self.field_named["second_half_field"]['segment'][2][0],
                self.field_named["second_half_field"]['segment'][2][1],
                self.field_named["second_half_field"]['segment'][3][0],
                self.field_named["second_half_field"]['segment'][3][1]
            ])
            if first_bottom!=None:
                    self.determined_topview_points.append(self.base_image_points["second_p_line_bottom"][0])
                    self.determined_frame_points.append([first_bottom[0],first_bottom[1]])    
        if self.field_named["second_mid_circle"]!={} and self.field_named["second_half_field"]!={} and  len(self.field_named["second_half_field"]['segment'])==4  :
            p1,p2 = self.calculate_parallel_second_line_points()
            first_bottom = self.find_intersection([
                p2[0],
                p2[1],
                p1[0],
                p1[1],
                self.field_named["second_half_field"]['segment'][0][0],
                self.field_named["second_half_field"]['segment'][0][1],
                self.field_named["second_half_field"]['segment'][1][0],
                self.field_named["second_half_field"]['segment'][1][1]
            ])
            if first_bottom!=None:
                    self.determined_topview_points.append(self.base_image_points["second_p_line_bottom"][1])
                    self.determined_frame_points.append([first_bottom[0],first_bottom[1]])    
    def error_calculation(self):
        try:
            coordinates = cv2.perspectiveTransform(np.array(self.determined_frame_points, dtype=np.float32).reshape(-1, 1, 2), np.linalg.inv(self.homography))
            distances = np.linalg.norm(coordinates - self.determined_topview_points, axis=-1)
            average_distance = np.mean(distances)
            mean_absolute_difference = np.sum(average_distance) / len(self.determined_topview_points)
            return mean_absolute_difference
        except:
            return 30