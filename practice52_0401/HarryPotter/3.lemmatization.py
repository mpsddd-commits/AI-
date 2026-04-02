import os
import ast
from nltk.stem import WordNetLemmatizer


# 1. 경로 직접 지정 (glob 없이 문자열로 설정)
input_path = 'word_total_cleaned/word_book_total.txt'
dest_folder = 'word_lemmatized_cleaned'
output_path = os.path.join(dest_folder, 'word_lemmatized_cleaned.txt')

# 표제어 추출기 객체 생성
lemmatizer = WordNetLemmatizer()

# 결과 폴더 생성
os.makedirs(dest_folder, exist_ok=True)

# 불용어(Stopwords) 리스트 (필요 시 추가)
stop_words = [] 

print(f"작업 시작: {input_path} 파일을 처리합니다.")

try:
    # 2. 파일 읽기
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 문자열 형태의 리스트['a', 'b']를 실제 파이썬 리스트 객체로 변환
        word_list = ast.literal_eval(content)
    
    print(f"파일 읽기 완료: 총 {len(word_list):,}개의 단어 로드됨.")

    # 3. Lemmatization (표제어 추출) 진행
    result = []
    for word in word_list:
        if word not in stop_words:
            # 단어의 원형(Lemma)을 찾아 리스트에 추가
            lemma = lemmatizer.lemmatize(word)
            result.append(lemma)
    
    # 4. 결과 저장 (리스트 형식 문자열로 저장)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(str(result))
        
    print("-" * 30)
    print(f"표제어 추출 완료! 파일 저장 경로: {output_path}")
    print(f"최종 처리된 단어 수: {len(result):,}개")

except FileNotFoundError:
    print(f"오류: '{input_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"처리 중 오류 발생: {e}")
