# from bs4 import BeautifulSoup
# import requests
from db import save

# url = "https://finance.naver.com"
# res = requests.get(url)

# print(res)

# soup = BeautifulSoup(res.text)
# print(soup.title)
# print(soup.title.text)

# print(soup.find("h1").text)

def etl():
  sql1 = f"""
            insert into db_to_air.`비행`
            SELECT * from db_air.`비행` where 년도 = 1987 and 월 = 10;
        """
  save(sql1)
  return True

result = etl()

if result:
  print("etl 실행 성공")
else:
  print("etl 실행 실패")