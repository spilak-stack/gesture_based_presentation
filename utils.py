import cv2
import numpy as np
import math
from func import *
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp

class Gaze:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
        self.face_mesh=self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # number of faces to track in each frame
            refine_landmarks=True,  # includes iris landmarks in the face mesh model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.results=None
        self.normalized_dist=None
        
    def eye_tracking(self,frame):
        self.results = self.face_mesh.process(frame)
        if self.results.multi_face_landmarks:
            return True

        else: 
            return False

    def gaze(self, frame):
        """
        The gaze function gets an image and face landmarks from mediapipe framework.
        The function draws the gaze direction into the frame.
        """

        points=self.results.multi_face_landmarks[0]
        '''
        2D image points.
        relative takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y) format
        '''
        image_points = np.array([
            relative(points.landmark[4], frame.shape),  # Nose tip
            relative(points.landmark[152], frame.shape),  # Chin
            relative(points.landmark[263], frame.shape),  # Left eye left corner
            relative(points.landmark[33], frame.shape),  # Right eye right corner
            relative(points.landmark[287], frame.shape),  # Left Mouth corner
            relative(points.landmark[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        '''
        2D image points.
        relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y,0) format
        '''
        image_points1 = np.array([
            relativeT(points.landmark[4], frame.shape),  # Nose tip
            relativeT(points.landmark[152], frame.shape),  # Chin
            relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
            relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
            relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
            relativeT(points.landmark[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (43.3, 32.7, -26),  # Right eye, right corner
            (-28.9, -28.9, -24.1),  # Left Mouth corner
            (28.9, -28.9, -24.1)  # Right mouth corner
        ])

        '''
        3D model eye points
        The center of the eye ball
        '''
        Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

        '''
        camera matrix estimation
        '''
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                    dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # 2d pupil location
        left_pupil = relative(points.landmark[468], frame.shape)
        right_pupil = relative(points.landmark[473], frame.shape)

        # Transformation between image point to world point
        _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

        if transformation is not None:  # if estimateAffine3D secsseded
            # project pupil image point into 3d world point 
            pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

            # 3D gaze point (10 is arbitrary value denoting gaze distance)
            S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

            # Project a 3D gaze direction onto the image plane.
            (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            # project 3D head pose into the image plane
            (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                            rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)
            # correct gaze for head rotation
            gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

            # Draw gaze line into screen
            p1 = (int(left_pupil[0]), int(left_pupil[1]))
            p2 = (int(gaze[0]), int(gaze[1]))


            dist= math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
            if p1[0]>p2[0]:
                dist=-dist

            if dist>=80:
                dist=80
            elif dist<=-80:
                dist=-80
            
            self.normalized_dist=640- int((dist+80)*4)

            return True
        
        else: 
            return False
class HandDetectorCVZone:
    def __init__(self, detection_con=0.8, max_hands=2):
        self.detector = HandDetector(detectionCon=detection_con, maxHands=max_hands)
        self.hands_data = None                  
        # hands_data = [
        # type(Right or Left), 
        # bbox list(x,y,w,h), 
        # fingersup list(0,0,0,0,0), 
        # landmark list(21 instance of [x,y])
        #   ]                      
        self.resized_hands_data = None          # 구조는 위와 동일하지만 landmark가 손 크기에 관계없이 일정하도록 resize함
        self.resized_hand_img=None
              
    def find_hands(self, img, draw=False):
        self.hands_data =[]
        new_img=img.copy()
        hands, draw_img = self.detector.findHands(new_img, draw=draw)
        if hands: 
            for hand in hands:
                hand_data=[
                    hand['type'],
                    hand['bbox'],
                    self.detector.fingersUp(hand),
                    list(map(lambda x: x[0:2],hand['lmList'])),
                ]
                self.hands_data.append(hand_data)
            self.extract_hand(img)

            return True
        else:
            self.hands_data=[]
            self.resized_hands_data=[]
            self.resized_hand_img=[]
            return False
        
    def extract_hand(self, img):
        """
        손이 있는 부분만 추출하여 고정 사이즈(500,500로 변경
        """
        self.resized_hand_img=[]
        self.resized_hands_data = []


        for hand_data in self.hands_data:
            bbox=[hand_data[1][0],hand_data[1][1],hand_data[1][2],hand_data[1][3]] # x,y,w,h

            if bbox[0]<0: bbox[0]=0
            if bbox[1]<0: bbox[1]=0
            aspect_ratio = bbox[2] / bbox[3]

            if aspect_ratio > 1:  # 가로가 더 긴 경우
                new_width = 500
                new_height = int(500 / aspect_ratio)
            else:  # 세로가 더 긴 경우
                new_width = int(500 * aspect_ratio)
                new_height = 500

            resized_hand_img = cv2.resize(img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], (new_width, new_height))
            canvas = np.zeros((500, 500, 3), dtype=np.uint8)
            x_offset = (500 - new_width) // 2
            y_offset = (500 - new_height) // 2
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_hand_img

            resized_hand_data=[
                    hand_data[0],
                    hand_data[1],
                    hand_data[2],
                ]

            resized_lm=[]
            for lm in hand_data[3]:
                cx, cy = lm
                resized_cx = int((cx - bbox[0]) / bbox[2] * new_width + x_offset)
                resized_cy = int((cy - bbox[1]) / bbox[3] * new_height + y_offset)
                resized_cx = min(max(resized_cx, 0), 499)
                resized_cy = min(max(resized_cy, 0), 499)

                resized_lm.append([resized_cx, resized_cy])
            resized_hand_data.append(resized_lm)
            self.resized_hands_data.append(resized_hand_data)

            self.resized_hand_img.append(canvas)

    def find_distance(self, hand_type:str, finger_num1:int, finger_num2:int):
        """
        한 손 내의 손가락 간의 거리를 계산하여 반환
        finger_num: 엄지~소지 0~4
        """
        finger_idx=(4, 8, 12, 16, 20) # 손가락 끝마디 번호

        for i, resized_hand_data in enumerate(self.resized_hands_data):
            if resized_hand_data[0] is hand_type:

                finger1_coor=self.resized_hands_data[i][3][finger_idx[finger_num1]]
                finger2_coor=self.resized_hands_data[i][3][finger_idx[finger_num2]]
                line_ = [finger1_coor[0] - finger2_coor[0], finger1_coor[1] - finger2_coor[1]]
                dist = math.sqrt(line_[0] ** 2 + line_[1] ** 2)

                return dist
            else: 
                return None
class VisionPT:
    def __init__(self, material_list:list) -> None:
        self.gesture=HandDetectorCVZone(detection_con=0.5)
        self.gaze=Gaze()

        self.material_list=material_list
        self.origin_material_list=material_list
        self.material_idx=0
        self.old_point=None
        self.new_point=None
        self.page_1_cnt=0
        self.page_10_cnt=0
        self.cnt=0
        self.draw_cnt=0
        self.draw_onoff_cnt=0
        self.keep_count=30

        self.init_material=self.material_list[self.material_idx].copy()        # 크롭 or 판서 등 계속 남겨둬야 하는 것은 이걸 수정
        self.init_note=np.ones_like(self.init_material)*255
        self.mode=[1,0,0] # pointer, crop, draw 순서


        self.crop_panels={
            'default' : cv2.flip(cv2.imread('./assets/crop/default.png',cv2.IMREAD_COLOR),1),
            'zoom' : cv2.flip(cv2.imread('./assets/crop/zoom.png',cv2.IMREAD_COLOR),1),
            'rotate' : cv2.flip(cv2.imread('./assets/crop/rotate.png',cv2.IMREAD_COLOR),1),
            'move' : cv2.flip(cv2.imread('./assets/crop/move.png',cv2.IMREAD_COLOR),1)
        }
        self.draw_panels={
            'default' : cv2.flip(cv2.imread('./assets/drawing/default.png',cv2.IMREAD_COLOR),1),
            'eraser' : cv2.flip(cv2.imread('./assets/drawing/eraser.png',cv2.IMREAD_COLOR),1),
            'pencil_black' : cv2.flip(cv2.imread('./assets/drawing/pencil_black.png',cv2.IMREAD_COLOR),1),
            'pencil_blue' : cv2.flip(cv2.imread('./assets/drawing/pencil_blue.png',cv2.IMREAD_COLOR),1),
            'pencil_red' : cv2.flip(cv2.imread('./assets/drawing/pencil_red.png',cv2.IMREAD_COLOR),1),
            'highlight_red' : cv2.flip(cv2.imread('./assets/drawing/highlight_red.png',cv2.IMREAD_COLOR),1),
            'highlight_yellow' : cv2.flip(cv2.imread('./assets/drawing/highlight_yellow.png',cv2.IMREAD_COLOR),1)
        }
        self.crop_panel=self.crop_panels['default']    
        self.draw_panel=self.draw_panels['default']


        self.cam=None

        self.current_material=self.init_material.copy()     # 수시로 갱신이 필요한 것은 이걸 수정
        self.current_note=self.init_note.copy()

        self.crop_mode=None
        self.old_crop_mode=None
        self.ref_vec=None

        self.draw_mode=None
        self.draw_coor=None
        
    def func_crop(self):
        self.current_material=self.init_material.copy()
        # self.current_note=self.init_note.copy()

        if len(self.gesture.hands_data)==2 and self.mode==[1,0,0]:
            if self.gesture.hands_data[0][2]==[0,1,0,0,0] and self.gesture.hands_data[1][2]==[0,1,0,0,0]:

                h, w, _ = self.current_material.shape
                ratio=h/w

                h2,w2,_ = self.cam.shape
                new_width=w2/1.5
                new_height=ratio*new_width
                range_point1= [int(w2/2-new_width/2), int(h2/2-new_height/2)]
                range_point2= [int(w2/2+new_width/2), int(h2/2+new_height/2)]

                overlay = self.cam.copy()
                cv2.rectangle(overlay, range_point1, range_point2, (0, 0, 0), -1)
                alpha=0.5
                self.cam = cv2.addWeighted(overlay, alpha, self.cam, 1 - alpha, 0, self.cam)  # 이미지 합성
                
                point1=self.gesture.hands_data[1][3][8]
                point2=self.gesture.hands_data[0][3][8]
                if point1[0]<point2[0]:
                    if point1[1]<point2[1]:     # point1,왼쪽 위
                        cv2.line(self.cam, pt1=point1, pt2=[point1[0]+30,point1[1]],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point1, pt2=[point1[0],point1[1]+30],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point2, pt2=[point2[0]-30,point2[1]],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point2, pt2=[point2[0],point2[1]-30],color=(255,255,255), thickness=2)
                    else:
                        cv2.line(self.cam, pt1=point1, pt2=[point1[0]+30,point1[1]],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point1, pt2=[point1[0],point1[1]-30],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point2, pt2=[point2[0]-30,point2[1]],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point2, pt2=[point2[0],point2[1]+30],color=(255,255,255), thickness=2)
                else:
                    if point1[1]<point2[1]:
                        cv2.line(self.cam, pt1=point1, pt2=[point1[0]-30,point1[1]],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point1, pt2=[point1[0],point1[1]+30],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point2, pt2=[point2[0]+30,point2[1]],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point2, pt2=[point2[0],point2[1]-30],color=(255,255,255), thickness=2)
                    else:
                        cv2.line(self.cam, pt1=point1, pt2=[point1[0]-30,point1[1]],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point1, pt2=[point1[0],point1[1]-30],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point2, pt2=[point2[0]+30,point2[1]],color=(255,255,255), thickness=2)
                        cv2.line(self.cam, pt1=point2, pt2=[point2[0],point2[1]+30],color=(255,255,255), thickness=2)

                if range_in(range_point1, range_point2, point1) and range_in(range_point1, range_point2, point2):
                    material_pt1 = transform_point(point1, range_point1, w, new_width)
                    material_pt2 = transform_point(point2, range_point1, w, new_width)

                    material_pt1 = flip_x_coordinate(material_pt1, w)
                    material_pt2 = flip_x_coordinate(material_pt2, w)

                    if material_pt1[0]<material_pt2[0]:
                        if material_pt1[1]<material_pt2[1]:     # point1,왼쪽 위
                            cv2.line(self.current_material, pt1=material_pt1, pt2=[material_pt1[0]+50,material_pt1[1]],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt1, pt2=[material_pt1[0],material_pt1[1]+50],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt2, pt2=[material_pt2[0]-50,material_pt2[1]],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt2, pt2=[material_pt2[0],material_pt2[1]-50],color=(0,0,0), thickness=3)
                        else:
                            cv2.line(self.current_material, pt1=material_pt1, pt2=[material_pt1[0]+50,material_pt1[1]],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt1, pt2=[material_pt1[0],material_pt1[1]-50],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt2, pt2=[material_pt2[0]-50,material_pt2[1]],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt2, pt2=[material_pt2[0],material_pt2[1]+50],color=(0,0,0), thickness=3)
                    else:
                        if material_pt1[1]<material_pt2[1]:
                            cv2.line(self.current_material, pt1=material_pt1, pt2=[material_pt1[0]-50,material_pt1[1]],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt1, pt2=[material_pt1[0],material_pt1[1]+50],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt2, pt2=[material_pt2[0]+50,material_pt2[1]],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt2, pt2=[material_pt2[0],material_pt2[1]-50],color=(0,0,0), thickness=3)
                        else:
                            cv2.line(self.current_material, pt1=material_pt1, pt2=[material_pt1[0]-50,material_pt1[1]],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt1, pt2=[material_pt1[0],material_pt1[1]-50],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt2, pt2=[material_pt2[0]+50,material_pt2[1]],color=(0,0,0), thickness=3)
                            cv2.line(self.current_material, pt1=material_pt2, pt2=[material_pt2[0],material_pt2[1]+50],color=(0,0,0), thickness=3)
                    
                    if self.old_point is None:
                        self.old_point=[point1, point2]
                        self.new_point=[point1, point2]
                    else:
                        self.old_point=self.new_point
                        self.new_point=[point1, point2]
                    
                        if distance(self.old_point[0], self.new_point[0])[1]<5 and distance(self.old_point[1], self.new_point[1])[1]<5:
                            self.cnt+=1
                        else:
                            self.cnt=0

                    if self.cnt>self.keep_count:
                        self.mode=[0,1,0]
                        crop_bbox=xyxy_to_xywh(material_pt1[0], material_pt1[1], material_pt2[0], material_pt2[1])
                        self.crop_img=self.init_material[crop_bbox[1]:crop_bbox[1]+crop_bbox[3], crop_bbox[0]:crop_bbox[0]+crop_bbox[2]]
                        self.current_crop_img=self.crop_img.copy()
                        # self.crop_img=cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2BGRA)
                        self.original_crop_shape=self.crop_img.shape[:2]
                        
                        self.origin_coor=[int(self.init_note.shape[1]/2), int(self.init_note.shape[0]/2)]          ##노트 한가운데로 세팅
                        self.img_locate()
                        self.cnt=0

        if self.mode==[0,1,0]:
            self.func_note()

    def vis_crop_panel(self):
        overlay = self.cam.copy()
        overlay[:60, :640]=self.crop_panel
        alpha=0.5
        self.cam = cv2.addWeighted(overlay, alpha, self.cam, 1 - alpha, 0, self.cam)  # 이미지 합성
                    
    def vis_draw_panel(self):
        overlay = self.cam.copy()
        overlay[:60, :640]=self.draw_panel
        alpha=0.5
        self.cam = cv2.addWeighted(overlay, alpha, self.cam, 1 - alpha, 0, self.cam)  # 이미지 합성

    def img_locate(self):
        """
        self.current_crop_img를 self.origin_coor을 중심으로 temp에 붙여넣는 함수입니다.
        self.origin_coor는 배경 이미지의 어디에든 위치할 수 있으며, 
        current_crop_img의 일부가 잘릴 수 있습니다.
        """
        temp = np.array(self.init_note)  # self.init_note를 복사합니다.

        # temp의 높이와 너비
        temp_height, temp_width = temp.shape[:2]

        # 붙여넣을 이미지의 높이와 너비
        crop_height, crop_width = self.current_crop_img.shape[:2]

        # x_offset, y_offset 계산 (중심 좌표 기준)
        y_offset = int(self.origin_coor[1] - crop_height / 2)
        x_offset = int(self.origin_coor[0] - crop_width / 2)

        # 붙여넣을 영역 계산 (temp 경계를 넘지 않도록)
        y_start = max(0, y_offset)
        x_start = max(0, x_offset)
        y_end = min(y_offset + crop_height, temp_height)
        x_end = min(x_offset + crop_width, temp_width)  # x_start 대신 x_offset 사용

        # 붙여넣을 이미지의 슬라이싱 범위 계산 (잘리는 부분 처리)
        crop_y_start = max(0, -y_offset)
        crop_x_start = max(0, -x_offset)
        crop_y_end = crop_y_start + (y_end - y_start)
        crop_x_end = crop_x_start + (x_end - x_start)  # x_start 대신 x_offset 사용

        # 알파 채널 처리
        alpha_channel = self.current_crop_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end, 3] / 255.0

        # 알파 채널을 이용하여 배경 이미지에 rgba 이미지 합성
        for c in range(3):
            temp[y_start:y_end, x_start:x_end, c] = (
                alpha_channel * self.current_crop_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end, c] +
                (1 - alpha_channel) * temp[y_start:y_end, x_start:x_end, c]
            )

        self.current_note = temp

    def img_fit(self):
        """
        self.current_crop_img를 self.origin_coor을 중심으로 temp에 붙여넣는 함수입니다.
        self.origin_coor는 배경 이미지의 어디에든 위치할 수 있으며, 
        current_crop_img의 일부가 잘릴 수 있습니다.
        """
        temp = np.array(self.init_note)  # self.init_note를 복사합니다.

        # temp의 높이와 너비
        temp_height, temp_width = temp.shape[:2]

        # 붙여넣을 이미지의 높이와 너비
        crop_height, crop_width = self.current_crop_img.shape[:2]

        # x_offset, y_offset 계산 (중심 좌표 기준)
        y_offset = int(self.origin_coor[1] - crop_height / 2)
        x_offset = int(self.origin_coor[0] - crop_width / 2)

        # 붙여넣을 영역 계산 (temp 경계를 넘지 않도록)
        y_start = max(0, y_offset)
        x_start = max(0, x_offset)
        y_end = min(y_offset + crop_height, temp_height)
        x_end = min(x_offset + crop_width, temp_width)  # x_start 대신 x_offset 사용

        # 붙여넣을 이미지의 슬라이싱 범위 계산 (잘리는 부분 처리)
        crop_y_start = max(0, -y_offset)
        crop_x_start = max(0, -x_offset)
        crop_y_end = crop_y_start + (y_end - y_start)
        crop_x_end = crop_x_start + (x_end - x_start)  # x_start 대신 x_offset 사용

        # 알파 채널 처리
        alpha_channel = self.current_crop_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end, 3] / 255.0

        # 알파 채널을 이용하여 배경 이미지에 rgba 이미지 합성
        for c in range(3):
            temp[y_start:y_end, x_start:x_end, c] = (
                alpha_channel * self.current_crop_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end, c] +
                (1 - alpha_channel) * temp[y_start:y_end, x_start:x_end, c]
            )

        self.init_note = temp

    def func_pointer(self):
        if self.mode==[1,0,0] and self.gesture.hands_data[0][0]=='Right' and self.gesture.hands_data[0][2]==[0,1,0,0,0]:
            h, w, _ = self.current_material.shape
            ratio=h/w

            h2,w2,_ = self.cam.shape
            new_width=w2/1.5
            new_height=ratio*new_width
            range_point1= [int(w2/2-new_width/2), int(h2/2-new_height/2)]
            range_point2= [int(w2/2+new_width/2), int(h2/2+new_height/2)]
            overlay = self.cam.copy()
            cv2.rectangle(overlay, range_point1, range_point2, (0, 0, 0), -1)
            alpha=0.5
            self.cam = cv2.addWeighted(overlay, alpha, self.cam, 1 - alpha, 0, self.cam)  # 이미지 합성
            
            point=self.gesture.hands_data[0][3][8]



            if range_in(range_point1, range_point2, point):
                material_pt = transform_point(point, range_point1, w, new_width)
                material_pt = flip_x_coordinate(material_pt, w)

                cv2.circle(self.current_material, center=material_pt, radius= 5, color=(0,0,255,255), thickness= -1)
                cv2.circle(self.current_material, center=material_pt, radius= 7, color=(0,0,0,255), thickness= 2)

    def func_note(self):
        self.vis_crop_panel()
        
        if self.gesture.hands_data[0][0] == 'Left':
            pt = flip_x_coordinate(self.gesture.hands_data[0][3][8], 640)
            if range_in((95,0),(155,60),pt):       # Zoom
                self.cnt+=1
                if self.cnt>self.keep_count:
                    self.crop_mode='Zoom'

            elif range_in((225,0),(285,60),pt):       # Rotate
                self.cnt+=1
                if self.cnt>self.keep_count:
                    self.crop_mode='Rotate'

            elif range_in((355,0),(415,60),pt):       # Rotate
                self.cnt+=1
                if self.cnt>self.keep_count:
                    self.crop_mode='Move'

            elif range_in((485,0),(545,60),pt):       # Rotate
                self.cnt+=1
                if self.cnt>self.keep_count:
                    self.crop_mode='Done'

            else: self.cnt=0

        
        if self.old_crop_mode is not None and self.old_crop_mode != self.crop_mode:
            self.crop_img=self.current_crop_img.copy()
            


        if self.crop_mode == 'Rotate':
            self.crop_panel=self.crop_panels['rotate']
            if len(self.gesture.hands_data)==2:
                if self.gesture.hands_data[0][2]==[0,1,0,0,0] and self.gesture.hands_data[1][2]==[0,1,0,0,0]:
                    if self.ref_vec is None:
                        self.ref_vec, _ = distance(self.gesture.hands_data[0][3][4], self.gesture.hands_data[1][3][4])
                    else:
                        new_vec, _ = distance(self.gesture.hands_data[0][3][4], self.gesture.hands_data[1][3][4])
                        angle_ = angle3(self.ref_vec, new_vec)
                        
                        self.current_crop_img=rotate(self.crop_img, angle_, self.original_crop_shape)
                        rows, cols = np.where(self.current_crop_img[:,:,3] != 0)

                        # 최소/최대 x, y 좌표 계산
                        x_min = np.min(cols)
                        x_max = np.max(cols)
                        y_min = np.min(rows)
                        y_max = np.max(rows)
                        self.current_crop_img=self.current_crop_img[y_min:y_max, x_min:x_max]
                        self.img_locate()
                        cv2.putText(self.current_note, text=str(-angle_), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0))



                else: self.ref_vec=None


        elif self.crop_mode == 'Zoom':
            self.crop_panel=self.crop_panels['zoom']
            if self.gesture.hands_data[0][0] == 'Right' and self.gesture.hands_data[0][2] == [1,1,0,0,0]:
                dist= self.gesture.find_distance('Right',0,1)
                dist=(dist-65)/300
                if dist>1:
                    dist=1
                elif dist<0:
                    dist=0

                dist=round(dist*2.5+0.5,1)
                
                self.current_crop_img = cv2.resize(self.crop_img, None, fx=dist, fy=dist)
                self.img_locate()
                cv2.putText(self.current_note, text='Zoom : ' + str(dist), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0))


        elif self.crop_mode == 'Move':
            self.crop_panel=self.crop_panels['move']

            if self.gesture.hands_data[0][0] == 'Right' and self.gesture.hands_data[0][2] == [0,1,0,0,0]:

                h, w, _ = self.current_note.shape
                ratio=h/w

                h2,w2,_ = self.cam.shape
                new_width=w2/1.5
                new_height=ratio*new_width
                range_point1= [int(w2/2-new_width/2), int(h2/2-new_height/2)]
                range_point2= [int(w2/2+new_width/2), int(h2/2+new_height/2)]

                overlay = self.cam.copy()
                cv2.rectangle(overlay, range_point1, range_point2, (0, 0, 0), -1)
                alpha=0.5
                self.cam = cv2.addWeighted(overlay, alpha, self.cam, 1 - alpha, 0, self.cam)  # 이미지 합성
                
                point=self.gesture.hands_data[0][3][8]
                        
                if range_in(range_point1, range_point2, point) :
                    note_pt = transform_point(point, range_point1, w, new_width)
                    note_pt = flip_x_coordinate(note_pt, w)

                    self.origin_coor=note_pt
                    self.img_locate()
                    

                
                

        elif self.crop_mode == 'Done':
            self.crop_panel=self.crop_panels['default']
            self.crop_mode=None
            self.mode=[1,0,0]
            self.old_crop_mode=None
            self.ref_vec=None
            self.img_fit()
            self.current_note=self.init_note



        self.old_crop_mode=self.crop_mode

    def func_drawing(self):
        if self.gesture.hands_data[0][0] == 'Left' and self.gesture.hands_data[0][2]==[0,0,0,0,0]:
            self.draw_onoff_cnt+=1
        else:
            self.draw_onoff_cnt=0
        
        if self.draw_onoff_cnt>self.keep_count:
            if self.mode==[1,0,0]: # 판서모드 진입
                self.mode=[0,0,1]
                self.draw_onoff_cnt=0

            elif self.mode==[0,0,1]: # 판서모드 해제
                self.mode=[1,0,0]
                self.draw_mode=None
                self.draw_panel=self.draw_panels['default']
                self.draw_onoff_cnt=0


        if self.mode==[0,0,1]:
            self.vis_draw_panel()
            self.highlight_layer=self.init_material.copy()

            if self.gesture.hands_data[0][0] == 'Left':
                pt = flip_x_coordinate(self.gesture.hands_data[0][3][8], 640)
                if range_in((57,0),(117,60),pt):       # Black
                    self.draw_cnt+=1
                    if self.draw_cnt>self.keep_count:
                        self.draw_mode='Black'
                        self.draw_panel=self.draw_panels['pencil_black']

                elif range_in((142,0),(202,60),pt):       # Red
                    self.draw_cnt+=1
                    if self.draw_cnt>self.keep_count:
                        self.draw_mode='Red'
                        self.draw_panel=self.draw_panels['pencil_red']

                elif range_in((227,0),(287,60),pt):       # Blue
                    self.draw_cnt+=1
                    if self.draw_cnt>self.keep_count:
                        self.draw_mode='Blue'
                        self.draw_panel=self.draw_panels['pencil_blue']

                elif range_in((362,0),(422,60),pt):       # Yellow_hl
                    self.draw_cnt+=1
                    if self.draw_cnt>self.keep_count:
                        self.draw_mode='Yellow_hl'
                        self.draw_panel=self.draw_panels['highlight_yellow']
                
                elif range_in((447,0),(507,60),pt):       # Red_hl
                    self.draw_cnt+=1
                    if self.draw_cnt>self.keep_count:
                        self.draw_mode='Red_hl'
                        self.draw_panel=self.draw_panels['highlight_red']
                
                elif range_in((553,0),(613,60),pt):       # Eraser
                    self.draw_cnt+=1
                    if self.draw_cnt>self.keep_count:
                        self.draw_mode='Eraser'
                        self.draw_panel=self.draw_panels['eraser']

                else: self.draw_cnt=0

            if self.gesture.hands_data[0][0]=='Right' and (self.gesture.hands_data[0][2]==[0,1,0,0,0] or self.gesture.hands_data[0][2]==[1,1,0,0,0]) and self.draw_mode is not None:

                h, w, _ = self.current_material.shape
                ratio=h/w

                h2,w2,_ = self.cam.shape
                new_width=w2/1.5
                new_height=ratio*new_width
                range_point1= [int(w2/2-new_width/2), int(h2/2-new_height/2)]
                range_point2= [int(w2/2+new_width/2), int(h2/2+new_height/2)]
                overlay = self.cam.copy()
                cv2.rectangle(overlay, range_point1, range_point2, (0, 0, 0), -1)
                alpha=0.5
                self.cam = cv2.addWeighted(overlay, alpha, self.cam, 1 - alpha, 0, self.cam)  # 이미지 합성
                
                point=self.gesture.hands_data[0][3][8]



                if range_in(range_point1, range_point2, point):
                    material_pt = transform_point(point, range_point1, w, new_width)
                    material_pt = flip_x_coordinate(material_pt, w)

                    if self.draw_mode == 'Black':
                        color=(0,0,0,255)
                        cv2.circle(self.current_material, center=material_pt, radius=5, color=color, thickness=-1)
                        cv2.circle(self.current_material, center=material_pt, radius=6, color=(0,0,0,0), thickness=1)

                    elif self.draw_mode == 'Red':
                        color=(0,0,255,255)
                        cv2.circle(self.current_material, center=material_pt, radius=5, color=color, thickness=-1)
                        cv2.circle(self.current_material, center=material_pt, radius=6, color=(0,0,0,0), thickness=1)
                    
                    elif self.draw_mode == 'Blue':
                        color=(255,0,0,255)
                        cv2.circle(self.current_material, center=material_pt, radius=5, color=color, thickness=-1)
                        cv2.circle(self.current_material, center=material_pt, radius=6, color=(0,0,0,0), thickness=1)
                    
                    elif self.draw_mode == 'Yellow_hl':
                        color=(0,255,255,50)
                        cv2.circle(self.current_material, center=material_pt, radius=15, color=color, thickness=-1)
                    
                    elif self.draw_mode == 'Red_hl':
                        color=(0,0,255,50)
                        cv2.circle(self.current_material, center=material_pt, radius=15, color=color, thickness=-1)
                    
                    elif self.draw_mode == 'Eraser':
                        color=(255,255,255,0)
                        cv2.circle(self.current_material, center=material_pt, radius=14, color=color, thickness=-1)
                        cv2.circle(self.current_material, center=material_pt, radius=15, color=(0,0,0,0), thickness=1)


                    if self.gesture.hands_data[0][2]==[0,1,0,0,0]:
                        if self.draw_coor is None:
                            self.draw_coor=material_pt
                        else:
                            if self.draw_mode == 'Eraser':
                                cv2.line(self.init_material, self.draw_coor, material_pt, color=color, thickness=15)
                            else:
                                if self.draw_coor != material_pt:
                                    if self.draw_mode[-2:]=='hl':
                                        cv2.line(self.highlight_layer, self.draw_coor, material_pt, color=color, thickness=15)
                                        self.init_material = cv2.addWeighted(self.highlight_layer, 0.5, self.init_material, 0.5, 0)
                                    else:
                                        cv2.line(self.init_material, self.draw_coor, material_pt, color=color, thickness=6)

                            self.draw_coor=material_pt
                    else: self.draw_coor=None
            
            if self.gesture.hands_data[0][0]=='Left' and self.gesture.hands_data[0][2]==[1,1,1,1,1]:
                self.init_material=self.origin_material_list[self.material_idx].copy()
                self.current_material=self.init_material.copy()

    def func_slide_10(self):
        if self.mode==[1,0,0] and self.gesture.hands_data[0][2]==[0,1,1,0,0]:
            cv2.putText(self.current_material, text=str(self.material_idx+1)+'/'+str(len(self.material_list)), 
                        org=(self.current_material.shape[1]-120,self.current_material.shape[0]-40), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0))
            self.page_10_cnt+=1
            if self.page_10_cnt>self.keep_count:
                if self.gesture.hands_data[0][0]=='Right':
                    self.slide_changed(10)

                elif self.gesture.hands_data[0][0]=='Left':
                    self.slide_changed(-10)

                self.page_10_cnt-=self.keep_count
        
        else: self.page_10_cnt=0

    def func_slide_1(self):
        if self.mode==[1,0,0] and self.gaze.gaze(self.cam) and self.gesture.hands_data[0][2]==[0,0,0,0,0] and self.gesture.hands_data[0][0]=='Right':
            gaze_pos=self.gaze.normalized_dist
            
            cv2.putText(self.current_material, text=str(self.material_idx+1)+'/'+str(len(self.material_list)), 
                        org=(self.current_material.shape[1]-120,self.current_material.shape[0]-40), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0))
            self.page_1_cnt+=1
            if self.page_1_cnt>self.keep_count:
                if gaze_pos >=480:
                    self.slide_changed(1)

                elif gaze_pos <=160:
                    self.slide_changed(-1)

                self.page_1_cnt-=self.keep_count

        else: self.page_1_cnt=0

    def slide_changed(self, page_num):
        self.material_list[self.material_idx]=self.init_material.copy()
        self.material_idx+=page_num
        if self.material_idx<0:
            self.material_idx=0
        elif self.material_idx>=len(self.material_list):
            self.material_idx=len(self.material_list)-1

        self.init_material=self.material_list[self.material_idx]
        self.current_material=self.init_material.copy()


            







