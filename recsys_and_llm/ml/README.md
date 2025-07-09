# UniCRS Multi-Turn Dialog 구현

이 프로젝트는 **UniCRS (Unified Conversational Recommender System)**에 Multi-Turn Dialog 기능을 추가하여 대화형 추천 시스템을 개선하는 것을 목표로 합니다. 이를 통해 대화 맥락(Context)을 기억하고, 사용자에게 더 개인화되고 정밀한 추천을 제공할 수 있습니다.

---

## 📝 배경

UniCRS는 대화형 추천 시스템(CRS) 아키텍처로, 사용자 입력(예: 영화 선호도, 장르)만으로 적절한 추천을 제공하는 Single-Turn 상호작용을 지원합니다.

이 프로젝트는 다음과 같은 기능을 추가하고자 합니다:
- 대화 맥락(Context History) 관리.
- Multi-Turn Dialog을 통해 사용자의 대화 흐름을 이해하고 반영.
- 과거 대화를 기반으로 정교한 추천 제공.

---

## 🚀 현재 코드 개요

현재 구현은 **RecWizard**라는 대화형 추천 시스템 도구를 사용합니다. 주요 구성 요소는 다음과 같습니다:

### 주요 구성 요소

1. **추천 모듈 (UnicrsRec)**:
   - 대화 맥락을 기반으로 영화 추천을 제공합니다.
   - 모델: `recwizard/unicrs-rec-redial`.

2. **생성 모듈 (UnicrsGen)**:
   - 사용자 입력에 대한 대화 응답을 생성합니다.
   - 모델: `recwizard/unicrs-gen-redial`.

3. **구성 (Configuration)**:
   - `FillBlankConfig`와 `ExpansionConfig`를 사용하여 파이프라인 동작 방식을 정의합니다.

4. **파이프라인 (Pipelines)**:
   - **FillBlankPipeline**: 대화에서 누락된 정보를 채우는 데 사용됩니다.
   - **ExpansionPipeline**: 사용자 입력을 확장하여 대화를 풍부하게 만듭니다.

5. **대화 관리**:
   - **Single-Turn 테스트**: `test_single_turn`.
   - **Multi-Turn 대화**: `chat_with_system`을 통해 사용자 입력을 저장하고 처리하는 방식으로 동작합니다.

---

## 📋 Issue #11 세부 내용

### 해야 할 일

1. **RecWizard 도구 이해**:
   - 관련 문서, 코드 구조, API를 검토.

2. **Multi-Turn Dialog 구현**:
   - 과거 대화를 기억하고 대화 흐름을 반영하도록 시스템 확장.

3. **대화 이력 (Context History) 관리**:
   - 사용자 입력 및 시스템 응답을 저장하고 활용할 수 있는 메커니즘 개발.
   - 예: `context` 리스트에 대화 기록 저장.

4. **Frontend/Backend 연동**:
   - CRS 백엔드와 UI 간의 원활한 상호작용 보장.

5. **모델 추가 학습**:
   - 프로젝트 요구사항에 맞춰 UniCRS 모델을 추가 학습 및 튜닝.

---

## 📌 실행 방법

### 환경 설정

1. 이 저장소를 클론합니다.
2. 종속성을 설치합니다:
   ```bash
   cd recwizard
   pip install -r .
   ```

3. main.ipynb를 실행합니다.
만약 torch 등 추가 의존성이 필요한 경우 설치바랍니다.