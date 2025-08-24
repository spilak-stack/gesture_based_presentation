import math
import cv2
import numpy as np
import os



relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def range_in(range_pt1, range_pt2, pt):
    if pt[0]> range_pt1[0] and pt[0] < range_pt2[0] and pt[1] > range_pt1[1] and pt[1] < range_pt2[1]:
        return True
    else:
        return False


def transform_point(point, origin, original_width, new_width):
    """
    점을 새로운 좌표계로 변환합니다.

    Args:
      point: 변환할 점의 좌표 (x, y).
      origin: 원본 좌표계의 시작점 (x, y).
      original_width: 원본 이미지의 가로 길이.
      new_width: 새로운 이미지의 가로 길이.

    Returns:
      새로운 좌표계에서의 점의 좌표 (x, y).
    """
    x = int((point[0] - origin[0]) * original_width / new_width)
    y = int((point[1] - origin[1]) * original_width / new_width)
    return (x, y)

def flip_x_coordinate(coor, image_width):
    """
    이미지상의 점을 좌우 반전하여 변환합니다.

    Args:
      coor: 변환할 점의 coor 좌표.
      image_width: 이미지의 가로 길이.

    Returns:
      좌우 반전된 점의 coor 좌표.
    """
    return [image_width - coor[0], coor[1]]

def distance(coor1, coor2):
    
    """
    선 벡터와 길이 반환
    """
    line_ = (coor1[0]-coor2[0], coor1[1]-coor2[1])
    dist = math.sqrt(line_[0]**2 + line_[1]**2)

    return line_, dist
def angle(coor1, coor2, coor3):
    """
    점 coor2에서 coor1, coor3을 각각 잇는 선분 사이의 각도
    (0~180 degree)
    """
    line1, length1 = distance(coor1, coor2)
    line2, length2 = distance(coor2, coor3)

    if length1 == 0 or length2 == 0:
        return 0 

    dot_product = line1[0]*line2[0] + line1[1]*line2[1]
    cos_theta = dot_product / (length1 * length2)
    cos_theta = max(-1, min(1, cos_theta)) 
    angle = math.degrees(math.acos(cos_theta))
    if abs(angle) > 180:
        angle = 360 - (abs)
    
    return angle

def angle2(line1, line2):
    """
    점 coor2에서 coor1, coor3을 각각 잇는 선분 사이의 각도
    (0~180 degree)
    """
    length1=math.sqrt(line1[0]**2 + line1[1]**2)
    length2=math.sqrt(line2[0]**2 + line2[1]**2)

    if length1 == 0 or length2 == 0:
        return 0 



    dot_product = line1[0]*line2[0] + line1[1]*line2[1]
    cos_theta = dot_product / (length1 * length2)
    cos_theta = max(-1, min(1, cos_theta)) 
    angle = math.degrees(math.acos(cos_theta))

    
    return angle

def angle3(line1, line2):
    """
    점 coor2에서 coor1, coor3을 각각 잇는 선분 사이의 각도
    (-180 ~ 180 degree)
    """
    length1 = math.sqrt(line1[0]**2 + line1[1]**2)
    length2 = math.sqrt(line2[0]**2 + line2[1]**2)

    if length1 == 0 or length2 == 0:
        return 0

    dot_product = line1[0]*line2[0] + line1[1]*line2[1]
    cos_theta = dot_product / (length1 * length2)
    cos_theta = max(-1, min(1, cos_theta)) 
    angle = math.degrees(math.acos(cos_theta))

    #외적을 이용하여 방향성 판별
    cross_product = line1[0] * line2[1] - line1[1] * line2[0]
    if cross_product < 0:
        angle = -angle

    angle=2*angle//5*5
    return angle

def xyxy_to_xywh(x1, y1, x2, y2):
    """xyxy 형식의 좌표를 xywh 형식으로 변환합니다.
    점의 위치 관계를 알 수 없는 경우에도 변환 가능합니다.

    Args:
      x1: 첫 번째 점의 x 좌표
      y1: 첫 번째 점의 y 좌표
      x2: 두 번째 점의 x 좌표
      y2: 두 번째 점의 y 좌표

    Returns:
      tuple: xywh 형식의 좌표 (x, y, w, h)
    """
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return (x, y, w, h)

def rotate(img, angle, original_shape):
    rows, cols = img.shape[:2]

    # 회전 중심 설정
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

    # 회전 후 이미지를 포함할 수 있는 새로운 이미지 크기 계산
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = original_shape[1]*3
    new_h = original_shape[0]*3

    # 회전 행렬 업데이트 (이동)
    M[0, 2] += (new_w / 2) - cols/2
    M[1, 2] += (new_h / 2) - rows/2

    # 이미지 회전, borderValue=(255, 255, 255)로 흰색 배경 지정
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(0, 0, 0, 0))

    return rotated_img

def read_materials(folder_path, resize_width):
    material_list=[]
    file_list = os.listdir(folder_path)
    for filename in file_list:
        path = os.path.join(folder_path, filename)
        material=cv2.imread(path,cv2.IMREAD_COLOR)
        ratio = material.shape[0]/material.shape[1]
        material=cv2.resize(material, (resize_width,int(ratio*resize_width)))
        material=cv2.cvtColor(material,cv2.COLOR_BGR2BGRA)
        material_list.append(material)

    
    return material_list





