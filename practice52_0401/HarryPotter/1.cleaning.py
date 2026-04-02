import re
import os
import glob


# 1. 경로 설정
src_folder = 'books'          # 원본 파일이 들어있는 폴더
dest_folder = 'first_cleaned'      # 결과물을 저장할 폴더


# 폴더가 없다면 생성 (전처리 파일을 저장할 폴더)
os.makedirs(dest_folder, exist_ok=True)

# 2. pages 폴더 내의 모든 .txt 파일 목록 가져오기
# glob을 사용하면 패턴에 맞는 파일을 리스트로 가져올 수 있습니다.
file_list = glob.glob(os.path.join(src_folder, 'Book*.txt'))

print(f"찾은 파일 개수: {len(file_list)}개")

for input_path in file_list:
    # 파일명만 추출 (예: Book1.txt)
    base_name = os.path.basename(input_path)

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"'{input_path}' 파일 읽기 성공.")

        # 원본 글자 수 저장
        original_count = len(text)
                             
        # 3. 클리닝 작업 수행
        # 맨 앞의 / 기호 및 공백 제거
        text = re.sub(r'^/\s*', '', text)

        # 페이지 하단 반복 문구 제거
        text = re.sub(r'Page\s*\|\s*\d+\s*Harry Potter and the Philosophers Stone\s*-\s*J\.K\.\s*Rowling', '', text)

        # 특수 문자 및 공백/소문자 정리
        cleaned_text = text.replace('“', '"').replace('”', '"').replace('—', '-')
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        cleaned_text = cleaned_text.lower()

        # 클리닝 후 글자 수 저장
        cleaned_count = len(cleaned_text)

        # 4. 결과 파일 저장 경로 지정 (cleaned/Book1_cleaned.txt)
        file_name_only = os.path.splitext(base_name)[0]
        output_path = os.path.join(dest_folder, f"{file_name_only}_cleaned.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        # 결과 출력 (파일명 | 원본 글자 수 -> 클리닝 후 글자 수)
        print(f"[{base_name}] 처리 완료")
        print(f"   - 원본 글자 수: {original_count:,}자")
        print(f"   - 정제 후 글자 수: {cleaned_count:,}자")
        print(f"   - 제거된 글자 수: {original_count - cleaned_count:,}자\n")
        

    except Exception as e:
        print(f"오류 발생: {e}")

print("모든 파일의 전처리가 완료되었습니다.")