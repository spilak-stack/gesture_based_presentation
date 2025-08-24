###########포인터#############################
# 오른손 검지로 동작

################페이지넘기기###################
# 오른손 검지+중지: +10페이지
# 왼손 검지+중지: -10페이지
# 오른손 주먹 든 채로 시선: +-1페이지

############크롭/회전/줌/이동##################
# 양손 검지로 영역 지정 후 유지하면 노트로 이미지가 넘어감
# 왼손 검지로 메뉴를 선택,유지하면 메뉴가 선택됨
# 줌은 오른손 엄지,검지로 0.5~3배까지 가능
# 회전은 양손 검지
# 이동은 오른손 검지
# 배치끝나면 메뉴에 체크 아이콘 선택,유지
# 배치가 끝난 이미지는 노트에 항시 유지되고 크롭 여러번 가능

##############판서############################
# 왼손 주먹 유지시 판서모드, 검정색 펜으로 기본 설정
# 왼손 검지로 메뉴를 선택,유지하면 펜이 선택됨
# 오른손 검지를 보이면 판서영역 활성화
# 오른손 검지를 편 상태에서 엄지를 접으면 써지고, 엄지를 펴면 안써짐
# 왼손 손바닥을 보여주면 클리어
# 왼손 주먹을 다시 유지하면 판서모드에서 벗어남

from utils import VisionPT
from func import read_materials
import cv2


material_list=read_materials(folder_path='./source', resize_width=1024)
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
vpt=VisionPT(material_list=material_list)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print('changed resolution width {} height {}'.format(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                                     cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret,frame=cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    vpt.cam=frame.copy()
    
    gesture_plag=vpt.gesture.find_hands(frame,draw=False)       # 필수, 탐지 성공하면 True반환
    gaze_plag=vpt.gaze.eye_tracking(frame)                      # 필수, 탐지 성공하면 True반환

    # 손을 탐지한 경우(왼손,오른손 구분 x)
    if gesture_plag:
        vpt.func_crop()
        vpt.func_drawing()
        vpt.func_pointer()
        vpt.func_slide_10()

        if gaze_plag:
            vpt.func_slide_1()

        
    
    cam_img=cv2.flip(vpt.cam,1)           # 좌우반전
    cv2.imshow('cam', cam_img)
    cv2.imshow('note ',vpt.current_note)	
    cv2.imshow('materials',vpt.current_material)
    # cv2.imshow('temp', vpt.origin_material_list[vpt.material_idx])


    if cv2.waitKey(5)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()