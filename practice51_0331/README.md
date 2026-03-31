prompt 

# Role
당신은 복잡한 텍스트 콘텐츠를 분석하여 지식 그래프(Knowledge Graph) 구조로 변환하는 **데이터 엔지니어링 및 NLP 전문가**입니다.

# Task
제공된 텍스트(또는 URL의 내용)를 바탕으로 주요 등장인물과 개체를 파악하고, 이를 지정된 JSON 형식의 중복되지 않은 **NODES** 리스트로 추출하십시오.

# extraction Rules
1. **Entity Identification**: 텍스트에서 인물, 생명체, 또는 중요한 개념을 추출합니다. (핵심 개체가 아닌 등장하는 모든 개체 추출)
2. **Labeling**: 
   - 작품의 세계관을 반영하여 리스트를 참고하여 카테고리를 분류합니다.
   - 반드시 한국어로 라벨링하십시오.
   - 라벨링 리스트 : ["인간", "포켓몬", "조직", "악당 단체", "지역", "대회", "현상"]
3. **Property Mapping**: `properties` 객체 안에 'name' 키를 포함하고, 값은 원문(영어) 또는 공식 명칭을 입력합니다.
4. **ID Generation**: 각 노드에 대해 'N0', 'N1', 'N2'와 같이 고유한 ID를 순차적으로 부여합니다.

# Output Format (JSON Only)
결과물은 오직 아래의 형식을 따르는 JSON 리스트여야 합니다:
[
  {"id": "N0", "label": "카테고리", "properties": {"name": "이름"}},
  ...
]

# Source Content
As Ash and his friends continue their journey to Sunyshore City, they are attacked by a wild Magnezone. As they reach a near-by town, they are informed by Officer Jenny that Magnezone and wild Metagross have suddenly appeared and is wreaking havoc throughout the town. As the group lead the Steel-types to the mountains using their Electric attacks, they meets up with Crispin, a mountain guard, who informs them that Magnezone and Metagross normally battle with each other on a regular basis to release the magnetism that builds up in their bodies, due to the strong magnetic forces emitted from the mountains. Recently, their battle arena (a deep crater) has become filled with water. Ash and co., Officer Jenny and Crispin have to find out why the crater is full of water, and a way to drain it, so Magnezone and Metagross can battle in peace away from the city.their fighting spirits and train hard for the rematch. Ash wins the rematch, and Buizel learns Ice Punch in the process.

# Result