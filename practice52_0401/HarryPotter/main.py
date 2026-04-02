import nltk
import os

def setFolder():
  folder_path = "C:/nltk_data"
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print("폴더를 생성했습니다.")
  else:
    print("폴더가 이미 존재합니다.")

if __name__ == '__main__':
  # setFolder()
  # print(nltk.__version__)
  # nltk.download()
  # nltk.download('punkt_tab')
  nltk.download('punkt')
  nltk.download('punkt_tab')