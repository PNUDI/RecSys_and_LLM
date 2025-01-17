# Front 프로젝트 README

## 프로젝트 개요
이 프로젝트는 Streamlit 기반의 대화형 추천 시스템 애플리케이션입니다. 
추천 시스템 모듈과 대화 관리 모듈을 통합하여 사용자가 보다 효과적으로 정보를 탐색하고 상호작용할 수 있도록 설계되었습니다.

---

## 디렉토리 구조
```
C:\USERS\USER\DESKTOP\24_WINTER\RECSYS_AND_LLM\FRONT
│  app.py
│  README.md
│  requirements.txt
│  utils.py
│  __init__.py
│  
├─assets
│      styles.css
│      
├─components
│  │  conversation_manager.py
│  │  pipeline_manager.py
│  │  recommend_main.py
│  │  response_generator.py
│  │  __init__.py
│  │  
│  └─__pycache__
│         
│          
├─pages
│  │  chat_page.py
│  │  components.py
│  │  main_page.py
│  │  __init__.py
│  │  
│  └─__pycache__
│          
│          
├─utils
│  │  image_utils.py
│  │  item_utils.py
│  │  session_utils.py
│  │  style_utils.py
│  │  __init__.py
│  │  
│  └─__pycache__
│          
│          
└─__pycache__
```

---

## 주요 파일 및 폴더 설명

### **1. 최상위 디렉토리**
- `app.py`: 애플리케이션의 진입점입니다. 로그인과 초기 설정을 관리합니다.
- `README.md`: 프로젝트에 대한 설명 및 가이드 문서입니다.
- `requirements.txt`: 프로젝트에서 사용하는 Python 라이브러리의 목록입니다.
- `utils.py`: 공통적으로 사용할 간단한 유틸리티 함수가 포함되어 있습니다.
- `__init__.py`: Python 패키지로 인식되도록 설정합니다.

---

### **2. assets**
- `styles.css`: 프로젝트의 전체적인 스타일을 정의한 CSS 파일입니다. Streamlit 앱의 UI 디자인을 개선하는 데 사용됩니다.

---

### **3. components**
- `conversation_manager.py`: 대화 내역 저장 및 로드와 관련된 기능을 처리합니다.
- `pipeline_manager.py`: 추천 파이프라인 설정 및 관리 기능을 담당합니다.
- `recommend_main.py`: 추천 결과의 메인 UI를 담당하는 모듈입니다.
- `response_generator.py`: 추천 시스템에서 생성된 응답 데이터를 처리하는 모듈입니다.
- `__init__.py`: 컴포넌트 모듈을 초기화하고 다른 모듈에서 쉽게 가져올 수 있도록 설정합니다.

---

### **4. pages**
- `chat_page.py`: 대화형 추천 시스템 페이지입니다. 사용자 입력과 모델 응답을 처리합니다.
- `components.py`: 페이지별 공통 UI 요소와 레이아웃을 정의합니다.
- `main_page.py`: 애플리케이션의 메인 페이지로, 추천 결과를 표시합니다.
- `__init__.py`: 페이지 모듈을 초기화하고 다른 모듈에서 쉽게 가져올 수 있도록 설정합니다.

---

### **5. utils**
- `image_utils.py`: 이미지 표시 및 렌더링과 관련된 유틸리티 함수가 포함되어 있습니다.
- `item_utils.py`: 추천 항목과 관련된 정보를 처리하는 유틸리티 함수가 포함되어 있습니다.
- `session_utils.py`: Streamlit 세션 상태 초기화 및 관리 기능을 담당합니다.
- `style_utils.py`: CSS 스타일을 로드하고 Streamlit 애플리케이션에 반영합니다.
- `__init__.py`: 유틸리티 모듈을 초기화하고 다른 모듈에서 쉽게 가져올 수 있도록 설정합니다.

---

## 실행 방법
1. **환경 설정**:
   - `Python 3.9+` 버전을 설치합니다.
   - `requirements.txt` 파일을 이용해 필요한 라이브러리를 설치합니다:
     ```bash
     pip install -r requirements.txt
     ```

2. **애플리케이션 실행**:
   - 다음 명령어를 사용해 Streamlit 애플리케이션을 실행합니다:
     ```bash
     streamlit run app.py
     ```

3. **로그인**:
   - 사용자 ID와 비밀번호를 입력해 로그인합니다.

4. **추천 결과 탐색**:
   - 대화형 추천 시스템을 통해 영화를 탐색하거나 추천받을 수 있습니다.

---

## 주요 의존성
- `Streamlit`: 웹 애플리케이션을 쉽게 빌드할 수 있도록 지원합니다.
- `Pillow`: 이미지 처리를 위한 라이브러리입니다.
- `requests`: HTTP 요청을 처리하기 위한 라이브러리입니다.
- `pyparsing`: 문자열 파싱 및 분석에 사용됩니다.