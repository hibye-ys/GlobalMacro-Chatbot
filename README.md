# 🤖 GlobalMacro-QA-Chatbot

### 🙏 Introduction
RAG system을 활용해 GlobalMacro 상황을 파악하고 과거와 비교해 적합한 자산군을 선택하기 위한 QA Chatbot 입니다.
관련된 데이터를 수집해 아래와 같은 과정을 거쳐 결론을 도출해냅니다.

<img width="1000" alt="스크린샷 2024-09-23 오전 2 32 51" src="https://github.com/user-attachments/assets/10524bd0-3f36-4c21-9960-86f0390dff5b">

----
### 🧲 Architecture
❊ 2024-09-23 기준 Agent 및 Supervisor, Judge는 구현되어 있지 않습니다.
<img width="1000" alt="스크린샷 2024-09-23 오전 2 34 43" src="https://github.com/user-attachments/assets/c604256a-6f44-4c12-abed-ba820580ba4e">

----

### 🧑‍💻 How To Use
아쉽게도 데이터 저작권 문제로 인해서 Global Chatbot을 직접적으로 사용할 수 없습니다.
본 프로젝트는 개인적인 사용용도와 포트폴리오용으로 수행한 프로젝트 입니다.
하지만 프로젝트를 수행하면서 겪은 시행착오와 만들어둔 코드들을 공유합니다.
프로젝트 과정은 제가 운영하는 [블로그](https://hibyeys.tistory.com/category/RAG%20Project?page=1)에서 확인가능하고 궁금한 내용은 블로그에 댓글로 남겨주세요.

#### 참고사항
1. 데이터 보호를 위해 파일 경로를 yaml파일로 관리하였습니다.
2. 코드를 사용하실때 config['settings']['base_path'] 등을 확인하고 yaml 파일을 만들어서 자신에게 맞게 적용해주세요. 
3. .env_sample을 참고해서 .env 파일에 API_KEY를 관리해주세요.
----


### 🏃‍➡️ Update Plan
- Searh Engine Tool을 활용해 Context를 보완 할 Agent 구성
- 질문에 맞게 대답이 잘 생성되었는지 판단하는 Judge 구성
- 정의한 5가지 관점으로 각 관점의 보고서를 작성하는 Multi-Agent 구성
- 각각의 Agent의 결과물을 평가하고 관리&감독하는 Supervisor Agent 구성
- Neo4j를 활용해 Graph RAG 테스트
