# RecSys_and_LLM

### 0. 프로젝트 설정 가이드

`RecSys_and_LLM` 프로젝트를 설정하고 실행하려면, 아래 지침을 따르시면 됩니다.

---

## 1. 저장소 클론

프로젝트를 로컬 머신 또는 서버에 클론합니다:

```bash
git clone https://github.com/PNUDI/RecSys_and_LLM.git
cd RecSys_and_LLM
```

---

## 2. Poetry 설치

Poetry는 Python 패키지 관리를 위한 도구입니다. 공식적으로 `pipx`를 통한 설치가 권장됩니다.

### pipx를 이용한 설치 (권장):

```bash
pip install --user pipx
pipx install poetry
```

위 방법이 안될 시에는 운영체제에 따라 아래와 같은 대체 방법을 사용해보세요!

- **macOS / Linux**:

  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```

- **Windows (PowerShell)**:

  ```powershell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
  ```

💡 Poetry는 가상 환경을 **자동으로 감지**하므로, Conda처럼 `activate`하지 않아도 됩니다.  
💡 또한, `poetry run`을 붙이면 명령어를 Poetry 환경에서 실행할 수 있습니다.  

---

## 3. 환경 변수 설정
설치 후, 환경 변수를 설정해야 합니다. 설치 경로를 확인하고 해당 경로를 시스템의 `PATH`에 추가하세요.

### Windows:
1. **설치 경로 확인**: 설치 완료 후 출력된 경로를 확인합니다. ex: `C:\Users\USERNAME\AppData\Roaming\Python\Scripts`
2. 환경 변수 설정:
   - **시스템 환경 변수 편집**: `제어판` > `시스템 및 보안` > `시스템` > `고급 시스템 설정` > `환경 변수`
   - **시스템 변수**에서 `Path` 선택 후 `편집`
   - `새로 만들기`를 클릭하고 위에서 확인한 경로를 추가
3. 변경 사항 적용을 위해 **시스템 재시작**

### Linux:

터미널에서 아래 명령어를 실행하여 `~/.bashrc`에 환경 변수를 추가합니다.

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. 설치 확인
설치가 완료되면 다음 명령어로 확인하세요:

```bash
poetry --version
```
Poetry를 설치한 후 `poetry --version`을 실행했을 때 **"command not found"** 오류가 발생한다면,  
환경 변수가 제대로 설정되지 않았을 수 있습니다.

---

## 5. Python 3.10.12 적용 (리눅스 서버)
이 프로젝트는 **Python 3.10.12 버전**에서 실행됩니다. (2025/02/10 기준)  
리눅스 서버에서 Poetry가 해당 버전을 사용하도록 설정해야 합니다.

### 5-1. Python 3.10.12 설치 여부 확인 및 설치 (pyenv 사용)
```bash
python3.10 --version
```
출력이 `Python 3.10.12`가 아니라면, `Python`이 올바르게 설치되지 않았거나 경로가 설정되지 않았을 수 있습니다. 

#### Linux에서 Python 3.10.12 설치

`Python 3.10.12`를 설치하려면 두 가지 방법 - **pyenv 또는 sudo (관리자 권한 필요)** - 이 있습니다. 

**1. pyenv 설치** (권장, 관리자 권한 없이 가능)
```bash
curl https://pyenv.run | bash
```
**2. 환경 변수 추가**
```bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc
```

**3. Python 3.10.12 설치 및 기본 버전 설정**
```bash
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.10.12
```

### 5-2. 기존 잘못된 Poetry 가상 환경 제거
Poetry가 다른 버전의 Python을 사용하고 있다면, 기존 가상 환경을 삭제합니다.

```bash
poetry env remove python
```

### 5-3. Python 3.10.12 적용
Python 3.10.12의 정확한 경로를 확인한 후, Poetry가 이를 사용하도록 설정합니다.

```bash
which python3.10  # Python 3.10.12의 경로 확인
poetry env use $(which python3.10)  # 해당 경로로 가상 환경 설정
```

설정된 가상 환경 확인 & 예시 사진:

```bash
poetry env info
```
![image](https://github.com/user-attachments/assets/e3a08054-3b86-4eea-86ff-726f802fec30)

---

## 6. 가상 환경 설정 및 의존성 설치

Poetry를 사용하여 의존성을 설치하고 가상 환경을 설정합니다:

```bash
poetry install
```

이 명령어는 `pyproject.toml`과 `poetry.lock`에 명시된 모든 의존성을 설치합니다.

#### 연구실 서버에서 poetry install 시 Pending 이슈 발생 시
keyring 관련 문제라면 아래 명령어를 실행하면 해결됩니다.

```bash
poetry config keyring.enabled false
poetry install
```

자세한 사항은 [서버 poetry 설치 무한 pending 이슈 해결방법](https://velog.io/@diluny/%EC%84%9C%EB%B2%84-poetry-%EC%84%A4%EC%B9%98-%EB%AC%B4%ED%95%9C-pending-%EC%9D%B4%EC%8A%88-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95) 참고.

---

## 7. Pre-commit 훅 설정 (develop)

코드 스타일과 포맷팅을 유지하기 위해 `pre-commit`을 설정합니다:

1. `pre-commit` 설치:

   ```bash
   poetry run pre-commit install
   ```

2. 훅 업데이트:

   ```bash
   poetry run pre-commit autoupdate
   ```

3. 모든 파일에 대해 수동으로 훅 실행:

   ```bash
   poetry run pre-commit run --all-files
   ```

---

## 8. 프로젝트 실행

**🥳 축하합니다!**  긴 과정을 따라오느라 고생 많으셨습니다.  
이제 가상 환경이 활성화된 상태에서 **다음과 같이 프로젝트를 실행**할 수 있습니다:

```bash
poetry run streamlit run recsys_and_llm/front/app.py
```

---

## A. 그 외 문제 해결 방법 🛠️

- **`pre-commit` 훅 또는 환경 관련 문제 발생 시**:

  ```bash
  poetry run pre-commit clean
  ```

- **의존성 설치 실패 시**: `pyproject.toml`에 명시된 Python 버전(`3.10.12`)과 현재 사용 중인 Python 버전이 일치하는지 확인해 보세요.


---

## B. 참고 사항 ⚠️

- 모든 의존성은 `poetry`를 통해 관리됩니다. **`pip`을 사용하여 패키지를 직접 설치하지 마세요.**
- 원활한 설정을 위해 **`poetry`가 전역적으로 설치되어 있는지 확인**하세요.  (위에서 설명한 방식을 따랐다면, 높은 확률로 괜찮을 것입니다.)


