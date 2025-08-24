import os

def rename_images(folder_path):
  """
  폴더 내에 있는 이미지 파일의 이름을 변경합니다.

  Args:
    folder_path: 이미지 파일이 있는 폴더 경로
  """
  image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')  # 이미지 확장자 목록
  file_list = os.listdir(folder_path)
  i = 1
  for filename in file_list:
    if filename.lower().endswith(image_extensions):
      src = os.path.join(folder_path, filename)
      dst = os.path.join(folder_path, f'image_{i:04d}{os.path.splitext(filename)[1]}')  # 새로운 파일명 형식 (예: image_0001.jpg)
      os.rename(src, dst)
      i += 1

# 폴더 경로를 지정합니다.
folder_path = './source'  # 실제 폴더 경로로 변경해야 합니다.

# 함수를 호출하여 이미지 파일의 이름을 변경합니다.
rename_images(folder_path)