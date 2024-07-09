import math
from sklearn.cluster import KMeans
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np


class Players:
    def __init__(self, tracker):
        self.players = []
        self.counter =0
        self.clustered = []
        self.player_mapping = {
             'original_trackers':[],
             'coordinates':[]
        }
        self.fitted_kmeans = False
        self.remapped_ids=[]
        self.details = []
        self.corresponding_ids ={}
        self.player_tracker = YOLO("apis/tactical_analysis/player_models/small_12-11_1088.pt")
        self.byte_tracker = tracker
        self.ball = []
        self.byte_tracker.removed_tracks=[]
        self.byte_tracker.lost_tracks=[]
        self.byte_tracker.tracked_tracks=[]
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.LEGACY
        )
        self.box_annotator = sv.BoxAnnotator(
            thickness=0,
            text_thickness=1,
            text_scale=0.75,
            color=sv.ColorPalette.LEGACY
        )
        self.ballfound = False
        self.kmeans = KMeans(n_clusters=2, random_state=0,n_init='auto')
        self.color_details = []
        self.coordinate =[]
        self.teams = []
        self.single_teams = {
            "team_1_x":[],
            "team_1_y":[],
            "team_2_x":[],
            "team_2_y":[]
        }

        self.new_trackers ={}

    def ballPosition(self,homography,frame):
        detected_ball={}
        self.ballfound = False
        for i in range(len(self.players.xyxy)):
            if self.players.class_id[i] == 0:
                self.ballfound = True
                x1,y1,x2,y2 = self.players.xyxy[i]
                coords =[round(((x1+x2)/2)),round(y2)]
                transformed = cv2.perspectiveTransform(np.array(coords, dtype=np.float32).reshape(-1, 1, 2), np.linalg.inv(homography)).reshape(-1, 2).tolist()
                ballDetections = sv.Detections(
                    xyxy = np.array([[x1,y1,x2,y2]]),
                    class_id = np.array([0]),
                )
                self.ellipse_annotator.annotate(scene=frame, detections=ballDetections)
                detected_ball["coordinates"] = list([int(item) for item in coords])
                detected_ball["Tcoordinates"] = list([int(item) for item in transformed[0]])
                self.ball.append(detected_ball)
                break

                    



    def detect_players(self,frame,homography):
        
        players_results = self.player_tracker.predict(source=frame,imgsz=1920)[0]
        self.color_details=[]
        self.coordinate=[]
        
        
        self.players =sv.Detections.from_ultralytics(players_results)
        self.ballPosition(homography,frame)
        self.players = self.byte_tracker.update_with_detections(self.players)
        tracker_ids=[]
        
        
        for i in range(len(self.players.xyxy)):
            
            
            if self.players.class_id[i] == 2:
                x1,y1,x2,y2 = self.players.xyxy[i]
                roi = frame[round(y1+((y2-y1)/3)):round(y2-((y2-y1)/3)), round(x1+((x2-x1)/5)):round(x2-((x2-x1)/5))]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue
                
                average_color = np.mean(roi, axis=(0, 1))
                self.color_details.append(average_color)
                self.coordinate.append([round(((x1+x2)/2)),round(y2)])
                tracker_ids.append(self.players.tracker_id[i])
        self.coordinate = cv2.perspectiveTransform(np.array(self.coordinate, dtype=np.float32).reshape(-1, 1, 2), np.linalg.inv(homography)).reshape(-1, 2).tolist()
        
        if self.player_mapping['original_trackers']==[]:
             self.player_mapping['original_trackers'] = tracker_ids
             self.player_mapping['coordinates'] = self.coordinate
        else:
            trackers = self.players.tracker_id.tolist()
            id_keys = [int(key) for key in self.corresponding_ids.keys()]
            for i in range(len(trackers)):
                for key in self.corresponding_ids:
                    if trackers[i] in self.corresponding_ids[key] :
                        if  key not in trackers and trackers[i] not in id_keys:
                            trackers[i] =int(key)
                        if trackers[i] in id_keys and trackers[i] in self.corresponding_ids[key]:
                            self.corresponding_ids[key].remove(trackers[i])

            for i in range(len(tracker_ids)):
                for key in self.corresponding_ids:
                    if tracker_ids[i] in self.corresponding_ids[key] :
                        if  key not in trackers and tracker_ids[i] not in id_keys:
                            tracker_ids[i] =int(key)
                        if tracker_ids[i] in id_keys and tracker_ids[i] in self.corresponding_ids[key]:
                            self.corresponding_ids[key].remove(tracker_ids[i])
            
            difference = list(set(self.player_mapping['original_trackers']) - set(tracker_ids))
            new_ids = list(set(tracker_ids) -set(self.player_mapping['original_trackers']))

            print(difference,new_ids)

            appended_ids = []
            appended_correspondance = []

            if len(difference)==0:
                for  i in new_ids:
                    if self.new_trackers.get(str(i)) == None:
                        self.new_trackers[i] = {
                            "count":1,
                            "coordinates":self.coordinate[tracker_ids.index(i)]
                        }
                    elif str(i) in self.new_trackers:
                        if self.new_trackers[str(i)].count==6:
                            self.player_mapping['original_trackers'].append(i)
                            self.player_mapping['coordinates'].append(self.new_trackers[str(i)].coordinates)
                            self.new_trackers.pop(str(i))
                        elif self.new_trackers[str(i)].count>6:
                            self.new_trackers.pop(str(i))
                        else:
                            self.new_trackers[str(i)].count+=1
                            self.new_trackers[str(i)].coordinates = self.coordinate[tracker_ids.index(i)]

            if difference!=[] and new_ids!=[]:
                distances = []
                ids =[]

                for j in difference:
                    missing_index = self.player_mapping['original_trackers'].index(j)
                    single_distacnes = []
                    single_ids =[]
                    for i in new_ids:
                        index = tracker_ids.index(i)
                        point1 = self.coordinate[index]
                        point2 = self.player_mapping['coordinates'][missing_index]
                        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                        added = False
                        if distance<=60:
                            for  distance_index in range(len(distances)):
                                if tracker_ids[index] in ids[distance_index]:
                                    if distance<distances[distance_index][ids[distance_index].index(tracker_ids[index])]:
                                        distances[distance_index].pop(ids[distance_index].index(tracker_ids[index]))
                                        ids[distance_index].pop(ids[distance_index].index(tracker_ids[index]))
                                        single_distacnes.append(distance)
                                        single_ids.append(tracker_ids[index])
                                        added = True
                                        break
                                    else:
                                        added = True
                                        break
                            if not added:
                                single_distacnes.append(distance)
                                single_ids.append(tracker_ids[index])


                    distances.append(single_distacnes)
                    ids.append(single_ids)
                size = len(distances)
                
                for i in range(size):
                     if len(distances[i])!=0:
                        index = distances[i].index(min(distances[i]))
                        if ids[i][index] not in appended_ids:
                            if str(difference[i]) in self.corresponding_ids:
                                if ids[i][index] not in self.corresponding_ids[str(difference[i])] and ids[i][index] !=difference[i] :
                                    self.corresponding_ids[str(difference[i])].append(ids[i][index])
                                    appended_ids.append(ids[i][index])
                                    appended_correspondance.append(difference[i])
                            else:
                                present_in_values=False
                                key_id = None
                                for key in self.corresponding_ids:
                                    if difference[i] in self.corresponding_ids[key]:
                                        present_in_values =True
                                        key_id = key
                                        break
                                
                                if present_in_values:
                                    self.corresponding_ids[key_id].append(ids[i][index])
                                    appended_ids.append(ids[i][index]) 
                                    appended_correspondance.append(difference[i])
                                else:
                                    self.corresponding_ids[str(difference[i])] =[ids[i][index]]
                                    appended_ids.append(ids[i][index]) 
                                    appended_correspondance.append(difference[i])


            for  i in new_ids:
                if i not in appended_ids:
                    if self.new_trackers.get(str(i)) == None:
                        self.new_trackers[i] = {
                            "count":1,
                            "coordinates":self.coordinate[tracker_ids.index(i)]
                        }
                    elif str(i) in self.new_trackers:
                        if self.new_trackers[str(i)].count==6:
                            self.player_mapping['original_trackers'].append(i)
                            self.player_mapping['coordinates'].append(self.new_trackers[str(i)].coordinates)
                            self.new_trackers.pop(str(i))
                        elif self.new_trackers[str(i)].count>6:
                            self.new_trackers.pop(str(i))
                        else:
                            self.new_trackers[str(i)].count+=1
                            self.new_trackers[str(i)].coordinates = self.coordinate[tracker_ids.index(i)]
        
            for value in range(len(appended_correspondance)):
                index = self.player_mapping['original_trackers'].index(appended_correspondance[value])
                self.player_mapping['original_trackers'][index] = appended_ids[value]
            for i in range(len(tracker_ids)):
                for index in range(len(self.player_mapping['original_trackers'])):
                    if tracker_ids[i] ==self.player_mapping['original_trackers'][index]:
                        self.player_mapping['coordinates'][index] = self.coordinate[i]

            remaining_ids = list(set(new_ids) - set(appended_ids))

            for i in range(len(remaining_ids)):
                index = tracker_ids.index(remaining_ids[i])
                self.player_mapping['original_trackers'].append(remaining_ids[i])
                self.player_mapping['coordinates'].append(self.coordinate[index])
            print(trackers)
            id_keys = [int(key) for key in self.corresponding_ids.keys()]
            for key in self.corresponding_ids:
                if key in trackers:
                    self.corresponding_ids.pop(key)
                
                for key2 in self.corresponding_ids:
                    if key2 !=key:
                        for value in self.corresponding_ids[key2]:
                            if value in self.corresponding_ids[key]:
                                if int(key)>int(key2):
                                    self.corresponding_ids[key].remove(value)
                                else:
                                    self.corresponding_ids[key2].remove(value)

            print(self.corresponding_ids)

            for i in range(len(trackers)):
                for key in self.corresponding_ids:
                    if len(self.corresponding_ids[key])!=0 and  min(self.corresponding_ids[key])<int(key):
                        self.corresponding_ids[key].remove(min(self.corresponding_ids[key]))
                    if trackers[i] in self.corresponding_ids[key] :
                        if  key not in trackers and trackers[i] not in id_keys:
                            trackers[i] =int(key)
                        if trackers[i] in id_keys and trackers[i] in self.corresponding_ids[key]:
                            self.corresponding_ids[key].remove(trackers[i])
            print(trackers)
            self.players.tracker_id = np.array(trackers)
        labels = [
            f"{tracker_id}"
            for _, confidence,confidence,mask, tracker_id, data
            in self.players
        ]
        frame = self.ellipse_annotator.annotate(scene=frame, detections=self.players)
        frame = self.box_annotator.annotate(scene=frame, detections=self.players, labels=labels)
            
        

    def clustering(self):
        try:
            if(len(self.color_details)!=0):
                self.details = []
                
                if not self.fitted_kmeans:
                    print("keams fitted")
                    self.kmeans.fit(np.array(self.color_details))
                    self.fitted_kmeans =True
                centers = self.kmeans.cluster_centers_
                trackers = []
                distance_to_ball = 80
                for i in range(len(self.players.class_id)):
                    if self.players.class_id[i] ==2:
                        trackers.append(self.players.tracker_id[i])
                        if len(self.ball)!=0 and self.ball[-1].get("tracker_id") is None:
                            x1,y1,x2,y2 = self.players.xyxy[i]
                            coords = [round(((x1+x2)/2)),round(y2)]
                            distance = math.sqrt((coords[0] - self.ball[-1]["coordinates"][0])**2 + (coords[1] - self.ball[-1]["coordinates"][1])**2)
                            if distance<math.sqrt((x2-x1)**2 + (y2-y1)**2)/2 or (distance_to_ball > distance and distance<=math.sqrt((x2-x1)**2 + (y2-y1)**2)/2+10):
                                self.ball[-1]["tracker_id"] = int(self.players.tracker_id[i])
                                self.ball[-1]["playerC"] = coords
                                distance_to_ball = distance
                        

                
                for i in range(len(self.coordinate)):
                    label = self.kmeans.predict(np.array(self.color_details[i]).reshape(1, -1)).tolist()
                    if  len(self.ball)!=0 and self.ball[-1].get("tracker_id") is not None and self.ball[-1].get("color") is None and int(trackers[i]) == self.ball[-1]["tracker_id"]:
                        self.ball[-1]["color"] = [int(item) for item in centers[label[0]].tolist()]
                        self.ball[-1]["playerCT"] = [int(item) for item in self.coordinate[i]]
                    player = {}
                    player["team"] = label[0]
                    player["color"] = [int(item) for item in centers[label[0]].tolist()]
                    player["coordinates"] = [int(item) for item in self.coordinate[i]]

                    player["tracker_id"] = int(trackers[i])
                    self.details.append(player)
        except Exception as e:
               print(e)
               self.details = []    

    def formation_images(self,frame):
        scale_y = 120 / frame.shape[1]
        scale_x = 80 / frame.shape[0]

        black_field1 = np.zeros((80, 120, 3), np.uint8)
        black_field2 = np.zeros((80, 120, 3), np.uint8)

        for i in range(len(self.details)):
            if self.details[i]["team"]==0:
                pt=tuple(self.details[i]["coordinates"])
                cv2.circle(black_field1, (int(pt[0]*scale_y), int(pt[1]*scale_x)), radius=3, color=(255,255,255), thickness=-1)
            else:
                pt=tuple(self.details[i]["coordinates"])
                cv2.circle(black_field2, (int(pt[0]*scale_y), int(pt[1]*scale_x)), radius=3, color=(255,255,255), thickness=-1)

        cv2.imwrite("team1.jpg",black_field1)
        cv2.imwrite("team2.jpg",black_field2)

        
                    
    def mark_players(self,frame,error):
        scale_y = 120 / frame.shape[1]
        self.team1 = []
        self.team2=[]
        scale_x = 80 / frame.shape[0]
        if self.counter==12:
                self.single_teams = {
                "team_1_x":[],
                "team_1_y":[],
                "team_2_x":[],
                "team_2_y":[]
                }
        for i in self.details:

            pt=tuple(i["coordinates"])
            cv2.circle(frame, (int(pt[0]), int(pt[1])), radius= int(error//2) if error < 60 else 30, color= (255, 255, 255), thickness=-1)
            cv2.circle(frame, (int(pt[0]), int(pt[1])), radius=10, color=i["color"], thickness=-1)
            if self.counter ==12:
                if i["team"] ==1:
                        self.single_teams["team_2_x"].append(int(pt[0]* scale_y))
                        self.single_teams["team_2_y"].append(int(pt[1]* scale_x))
                else:
                        self.single_teams["team_1_x"].append(int(pt[0]* scale_y))
                        self.single_teams["team_1_y"].append(int(pt[1]* scale_x))

        if self.counter==12:
            self.counter=0
        else:
            self.counter+=1

