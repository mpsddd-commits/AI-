import os
import glob
from tensorflow.keras.preprocessing.text import text_to_word_sequence
# from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import ast  # 문자열 형태의 리스트를 실제 리스트로 변환하기 위해 필요
# 문장 별 추출
def step0():
    # 1. 경로 설정
    src_folder = 'first_cleaned'          # 원본 파일이 들어있는 폴더
    dest_folder = 'sent_token_cleaned'        # 결과물을 저장할 폴더

    # 폴더가 없다면 생성 (전처리 파일을 저장할 폴더)
    os.makedirs(dest_folder, exist_ok=True)

    # 2. pages 폴더 내의 모든 .txt 파일 목록 가져오기
    # glob을 사용하면 패턴에 맞는 파일을 리스트로 가져올 수 있습니다.
    file_list = glob.glob(os.path.join(src_folder, 'Book*.txt'))

    print(f"찾은 파일 개수: {len(file_list)}개")

    for input_path in file_list:
    # 파일명만 추출 (예: Book1_cleaned.txt)
        base_name = os.path.basename(input_path)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 3. 문장 단위 토큰화 (sent_tokenize)
            sentences = sent_tokenize(text)
                
            # 실수버젼    결과물 생성 (문장별로 줄바꿈 추가)
            # 문장 리스트를 줄바꿈 문자(\n)로 합쳐서 하나의 텍스트로 만듭니다.
            # sentence_text = '\n'.join(sentences)
                            
            # 4. 저장 경로 설정 및 저장
            output_path = os.path.join(dest_folder, base_name)

            # 5. 리스트 형식 문자열로 저장 (수정된 부분)
            with open(output_path, 'w', encoding='utf-8') as f:
                # 리스트 객체를 str()로 감싸면 ['문장1', '문장2'] 형태로 저장됩니다.
                f.write(str(sentences))    
            # with open(output_path, 'w', encoding='utf-8') as f:
            #         f.write(sentence_text)
                
            print(f"'{base_name}' 처리 성공: {len(sentences):,}개 문장 추출 완료.")

        except Exception as e:
            print(f"파일 {base_name} 처리 중 오류 발생: {e}")

# tensorflow로 단어 토큰화
def step1():
    # 1. 경로 설정
    src_folder = 'sent_token_cleaned'               # 원본 파일이 들어있는 폴더
    dest_folder = 'word_token_cleaned'        # 결과물을 저장할 폴더

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
            
            # 3. 단어 단위 토큰화 (sent_tokenize)
            word = text_to_word_sequence(text)
                         
            # 4. 저장 경로 설정 및 저장
            output_path = os.path.join(dest_folder, base_name)
                
            # with open(output_path, 'w', encoding='utf-8') as f:
            #         word_text = ' '.join(word) 
            #         f.write(word_text)
            
            # 5. 리스트 형식 문자열로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                # word 리스트를 문자열로 변환하여 저장
                f.write(str(word))
                
            print(f"'{base_name}' 처리 성공: {len(word):,}개 단어 추출 완료.")

        except Exception as e:
            print(f"파일 {base_name} 처리 중 오류 발생: {e}")

# 문장토큰 통합본 만들기
def step2():

    # 1. 경로 설정
    src_folder = 'sent_token_cleaned'               # 원본 파일이 들어있는 폴더
    dest_folder = 'sent_total_cleaned'        # 결과물을 저장할 폴더
    output_filename = 'sent_book_total.txt'

    # 폴더 생성
    os.makedirs(dest_folder, exist_ok=True)

    # 2. 파일 목록 가져오기 (정렬하여 순서대로 합치기 위해 sorted 사용)
    file_list = sorted(glob.glob(os.path.join(src_folder, 'Book*.txt')))

    if not file_list:
        print(f"'{src_folder}' 폴더에 처리할 파일이 없습니다.")
        return

    print(f"총 {len(file_list)}개의 파일을 하나로 합칩니다.")

    all_text = []

    # 3. 각 파일의 내용을 읽어서 리스트에 추가
    for input_path in file_list:
        base_name = os.path.basename(input_path)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
                all_text.append(text)
            print(f"{base_name} 파일을 읽었습니다.")
        except Exception as e:
            print(f"파일 '{base_name}' 처리 중 오류 발생: {e}")

    # 4. 하나의 파일로 저장
    output_path = os.path.join(dest_folder, output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(str(all_text))
        print("-" * 30)
        print(f"통합 완료! 파일 경로: {output_path}")
        print(f"최종 파일 크기: {os.path.getsize(output_path):,} bytes")
    except Exception as e:
        print(f"통합 파일 저장 중 오류 발생: {e}")

# 단어토큰 통합본 만들기
def step3():

    # 1. 경로 설정
    src_folder = 'word_token_cleaned'               # 원본 파일이 들어있는 폴더
    dest_folder = 'word_total_cleaned'              # 결과물을 저장할 폴더
    output_filename = 'word_book_total.txt'

    # 폴더 생성
    os.makedirs(dest_folder, exist_ok=True)

    # 2. 파일 목록 가져오기 (정렬하여 순서대로 합치기 위해 sorted 사용)
    file_list = sorted(glob.glob(os.path.join(src_folder, 'Book*.txt')))

    if not file_list:
        print(f"'{src_folder}' 폴더에 처리할 파일이 없습니다.")
        return

    print(f"총 {len(file_list)}개의 파일을 하나로 합칩니다.")

    total_word_list = []

    # 3. 각 파일의 내용을 읽어서 리스트에 추가
    for input_path in file_list:
        base_name = os.path.basename(input_path)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 문자열 형태인 "['word1', 'word2']"를 실제 파이썬 리스트 객체로 변환
                current_list = ast.literal_eval(content)
                # 리스트끼리 더하여(extend) 하나의 큰 리스트로 만듦
                total_word_list.extend(current_list)
            print(f"{base_name} 파일을 읽고 리스트에 추가했습니다. (단어 수: {len(current_list):,}개)")
        except Exception as e:
            print(f"파일 '{base_name}' 처리 중 오류 발생: {e}")

    # 4. 하나의 파일로 저장
    output_path = os.path.join(dest_folder, output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(str(total_word_list))

        print("-" * 30)
        print(f"통합 완료! 파일 경로: {output_path}")
        print(f"최종 파일 크기: {os.path.getsize(output_path):,} bytes")
    except Exception as e:
        print(f"통합 파일 저장 중 오류 발생: {e}")


# 함수 실행
if __name__ == "__main__":
    step0()
    step1()
    step2()
    step3()

