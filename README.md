<p align="center">
  <img src="docs/logo.svg" width="72" alt="Fovea" />
</p>
<h1 align="center">Fovea</h1>
<p align="center"><b>Local-first computer-vision dataset review &amp; labeling workspace.</b><br/>
YOLO 포맷 데이터셋을 등록하면 인덱싱 → 데이터 품질 검사 → 갤러리 검수 → 라벨링(AI 제안 포함) → 평가 → 내보내기까지 한 화면에서. 업로드도, 계정도 없습니다.</p>

---

> 이전 이름은 `cv-data-viewer`였습니다. 옛 URL은 GitHub가 자동으로 리다이렉트하지만, 기존 클론은 원격 주소를 갱신해 주세요:
> `git remote set-url origin https://github.com/jihunz/fovea.git`

## 왜 Fovea인가

망막에서 시력이 가장 예리한 지점이 **fovea(중심와)** 입니다. Fovea는 데이터셋을 가장 자세히 들여다보는 도구를 지향합니다.

- **한 번 등록하면 끝** — 폴더·`data.yaml`·`train.txt` 중 무엇을 지정해도 레이아웃을 감지하고 SQLite 인덱스(이미지·라벨·박스·해상도)를 만듭니다. 45k 장도 서버 페이지네이션과 썸네일 캐시로 즉시 응답합니다.
- **모델 없이 찾는 데이터 문제** — 근중복·분할 간 누수·시퀀스 공백·이상한 박스 모양·흐리거나 어두운 이미지를 인덱싱 직후 자동으로 검사하고, 라벨을 고치면 몇 초 안에 다시 검사합니다.
- **AI는 제안만 한다** — 로컬 YOLO/YOLOE/SAM 또는 Claude·Gemini·OpenAI·OpenAI 호환 서버의 결과는 기본적으로 점선 *제안*으로 저장되고, 사람이 키 하나로 수락·거절합니다. 일괄 실행은 비용을 미리 보여 주고 클라우드 호출은 Settings에서 정한 30일 지출 상한에 닿으면 멈추며, API 키는 서버 밖으로 나가지 않습니다.
- **라벨 파일은 안전하게** — 편집한 파일만 원자적으로 다시 쓰고, 일괄 작업 전에는 자동 스냅샷을 남겨 되돌릴 수 있습니다.
- **오래 봐도 덜 피곤하게** — 이미지는 페이드 없이 즉시 바뀌고, 깜박이는 표시와 레이아웃 흔들림이 없습니다([docs/visual-comfort.md](docs/visual-comfort.md)).

## 빠른 시작

```bash
git clone https://github.com/jihunz/fovea.git && cd fovea
```

### 로컬 (Python 3.11+)

```bash
cd src
pip install -r requirements.txt        # 전체 설치: 핵심 + 선택 패키지(numpy, ultralytics/torch, anthropic, httpx, tifffile)
python -m fovea --port 8000            # http://localhost:8000
```

가볍게 쓰려면 핵심 패키지만 설치해도 됩니다: `pip install "fastapi>=0.115.6,<0.144" "starlette>=0.40,<1.9" "uvicorn[standard]>=0.31,<0.55" "Jinja2>=3.1" "Pillow>=10.3" "PyYAML>=6.0"`. 이때 AI 공급자는 모두 쓸 수 없습니다: 로컬 모델(YOLO/YOLOE/SAM)은 `ultralytics`·`torch`, Claude는 `anthropic`, Gemini·OpenAI·OpenAI 호환 서버는 `httpx`가 필요합니다. Settings와 AI 탭의 상태 줄(키가 없는 클라우드 공급자는 키를 넣은 뒤)과 **Test**가 빠진 패키지의 `pip install` 명령을 알려 줍니다. 라벨링·검수·데이터 품질·내보내기는 그대로 동작합니다. 선택 패키지의 역할은 `requirements.txt` 주석에 있습니다. iPhone HEIC 사진은 `pip install pillow-heif`.

| 옵션 | 설명 |
| --- | --- |
| `--host` | 기본 `127.0.0.1`(이 PC에서만 접속). `0.0.0.0`이면 다른 PC도 접속 — 계정이 없으므로 `FOVEA_TOKEN=<비밀값>`을 함께 설정하세요 |
| `--port` | 기본 8000 |
| `--data-dir` | 인덱스 DB·썸네일·크롭 캐시·API 키 파일 위치 (기본 `src/data`) |
| `--model-dir` | 모델 가중치 폴더 (기본 `src/model`) |
| `--allowed-hosts` | IP·localhost·이 PC 이름 외에 접속을 허용할 호스트 이름(쉼표 구분, `*.lab.example` 가능) |
| `--allow-remote-ai` | 다른 PC에서도 이 서버에 저장된 키로 클라우드 AI를 쓰도록 허용 (`FOVEA_ALLOW_REMOTE_AI=1`과 같음) |
| `--reload` | 개발용 자동 재시작 |

`Ctrl+C`로 멈추면 진행 중인 잡을 취소하고 5초 안에 깔끔하게 종료합니다. API 문서: `http://localhost:8000/api/docs`.

### Docker

```bash
cd src
cp .env.example .env                   # HOST_PATH = 데이터셋이 있는 폴더 (컨테이너의 /host 로 마운트)
docker compose up --build -d           # http://localhost:8888
```

- 기본적으로 이 PC에서만 접속되도록 `127.0.0.1:8888`에 포트를 엽니다. 이 PC의 브라우저는 Docker 브리지 주소로 들어오지만 `FOVEA_LOCAL_NETS`가 그 주소를 "이 PC"로 취급하므로 토큰 없이, 클라우드 AI도 그대로 씁니다.
- **같은 네트워크의 다른 PC에 열기**: `docker-compose.yml`의 포트를 `"8888:8000"`으로 바꾸고 `.env`에 `FOVEA_TOKEN`을 설정한 뒤 `FOVEA_LOCAL_NETS` 줄을 지우세요. 이제 컨테이너에는 모든 접속이 Docker 브리지 주소(예: `172.17.0.1`)로 보여 이 PC와 다른 PC를 구별할 수 없으므로 **Docker를 실행한 PC의 브라우저도 다른 PC로 취급됩니다**. 이 PC에서도 토큰을 입력해야 하고(브라우저가 30일 기억), 클라우드 AI 호출과 API 키·클라우드 공급자 설정 변경은 403으로 거부됩니다(YOLO·YOLOE·SAM은 그대로). 클라우드 AI가 필요하면 `.env`에 `FOVEA_ALLOW_REMOTE_AI=1`을 넣고 `docker compose up -d`로 컨테이너를 다시 만드세요. 그러면 토큰을 아는 모든 PC가 저장된 키로 클라우드 AI를 쓰므로 Settings에서 공급자별 지출 상한을 걸어 두세요. 클라우드 AI를 이 PC에서만 쓰려면 Docker 대신 `python -m fovea --host 0.0.0.0`으로 실행하고, 이 PC에서는 `http://localhost:8000`으로 접속하세요(토큰 없이, 클라우드 AI 그대로).
- `.env`의 값은 `docker-compose.yml`의 `environment:`에 나열된 변수만 컨테이너로 전달됩니다: `FOVEA_TOKEN`, `FOVEA_ALLOWED_HOSTS`, `FOVEA_ALLOWED_ORIGINS`, `FOVEA_ALLOW_REMOTE_AI`, 세 API 키(`HOST_PATH`·`INSTALL_AI`·`UID`/`GID`는 compose가 마운트·빌드·실행 사용자에 씁니다). 그 밖의 변수(예: `FORWARDED_ALLOW_IPS`, `FOVEA_THUMB_CACHE_MAX_GB`)는 `.env`에 적어도 전달되지 않으니 `environment:`에 `- 이름=값` 줄로 추가하세요.
- 컨테이너는 root가 아닌 사용자(`.env`의 `UID`/`GID`)로 실행되어, 데이터셋 폴더에 쓴 라벨·내보내기 파일의 소유자가 바뀌지 않습니다.
- AI 패키지 없이 가볍게 빌드: `INSTALL_AI=0 docker compose up --build -d`. 클라우드 키는 셸이나 `.env`에 `ANTHROPIC_API_KEY`·`GEMINI_API_KEY`·`OPENAI_API_KEY`가 있을 때만 컨테이너로 전달됩니다(이미지에 기록되지 않음).
- 호스트 경로(`/Users/me/data/...`, `C:\data\...`)를 그대로 입력하면 컨테이너 경로로 자동 매핑됩니다(아직 없는 내보내기 폴더도). 인덱스 DB·캐시는 `src/data/`, 모델 가중치는 `src/model/`에 두며 둘 다 이미지에 포함되지 않고 마운트됩니다.
- macOS의 Docker Desktop에서는 바인드 마운트한 `./data` 폴더에서 파일 잠금이 프로세스 사이에 공유되지 않습니다. Fovea는 2,000장 이상인 데이터셋을 처음 검사할 때 이를 확인하고, 그런 폴더에서는 데이터 품질 검사를 별도 프로세스 대신 서버 안에서 계산합니다(인덱스는 안전하고, 로그에 경고가 한 줄 남습니다. 큰 데이터셋을 검사하는 동안 다른 화면이 조금 느려질 수 있습니다).

## 화면별 기능

### Home · Add dataset
폴더, `data.yaml`, `train.txt` 중 아무거나 지정하면 입력하는 동안 레이아웃을 감지해 분할별 이미지·라벨 폴더를 미리 보여 줍니다. 지원 레이아웃: `images/<split>`+`labels/<split>`, `<split>/images`+`labels`, `images/`+`labels/`, 이미지와 txt가 섞인 폴더, 라벨 없는 폴더(라벨 폴더 자동 생성), `data.yaml`(Roboflow `../train/images` 경로 포함), `train.txt`, CVAT YOLO 1.1 내보내기. 분할은 train, val, test, 그다음 나머지(자연 정렬) 순서로 표시됩니다. 여러 데이터셋이 든 폴더를 지정하면 그중 하나를 고르는 버튼이 나옵니다. 버튼은 경로 칸만 바꾸므로 고른 뒤 **Register & index**를 누르고, 데이터셋마다 반복합니다. 이미 등록된 폴더는 *Open “이름”* / *Register anyway* 선택지로 안내합니다. 첫 인덱싱을 취소하면 *Partial index* 상태가 되고 **Resume indexing**으로 이어서 합니다.

**폴더를 옮겼을 때**: 데이터셋 폴더를 옮겼거나, 드라이브 이름이 바뀌었거나, 다른 PC·Docker에서 만든 인덱스라 파일을 찾지 못하면 데이터셋 화면에 *These files are not on this machine*이 뜹니다. 그 안내나 Home·데이터셋 메뉴의 **Move to a new path…**로 새 위치(폴더, `data.yaml`, `train.txt`)를 지정하면 같은 데이터셋으로 다시 인덱싱합니다. 데이터셋 ID와 **이미지 ID가 그대로 유지**되므로 리뷰 표시, AI 제안과 감사 소견, 데이터 품질 플래그, 스냅샷, 클래스 카드의 예시 박스, 작업 묶음(`?set=`)이 모두 따라옵니다. 레이아웃이 달라져 상대 경로가 바뀐 파일도 리뷰 표시와 같은 규칙으로 짝을 찾아 옮깁니다. 기다리던 Economy 배치는 끊지 않고, 다시 인덱싱이 끝나면 이어서 결과를 받습니다.

### Overview
- **KPI 타일**: Images · Labeled · Unlabeled · Boxes · Classes · Label issues · Reviewed. 타일을 누르면 해당 이미지로 필터된 Explore로(Boxes는 Objects로) 이동합니다. Classes 타일은 정보만 표시합니다.
- **Health check**: 라벨 파일 문제(형식 오류, 클래스 ID 오류, 범위 밖·면적 0·미세·중복 박스)와 클래스 문제(박스 없는 클래스, 박스 수가 클래스별 박스 수 중앙값의 10% 미만인 클래스, 어떤 분할에 없는 클래스).
- **Data quality**: 근중복·분할 간 누수·라벨이 다른 쌍둥이 이미지·시퀀스 공백/튐/깜박임/개수 급변·이상한 박스 모양·흐림/어두움/과노출을 그룹별로 세고 **Review**로 해당 이미지만 가장 의심스러운 것부터(플래그 점수 순) 엽니다. 누수는 내보내기가 폴더를 매핑하는 분할(train/val/test) 기준이라, 같은 분할로 가는 두 폴더(`test`와 `test-1854-1860`)에 같은 프레임이 있으면 누수가 아니라 정보성 *Same frame in two folders of one split*로 셉니다. 여러 분할에 걸친 시퀀스(같은 영상의 프레임이 학습·평가에 함께 들어간 경우)도 나열합니다. 인덱싱이 끝나면 자동으로 실행되고, 진행 상황은 카드 안에 실시간으로 표시됩니다. 2,000장 이상인 데이터셋은 검사를 잠깐 띄운 별도 프로세스에서 계산하므로 검사 중에도 다른 화면이 느려지지 않습니다. 자세한 기준: [docs/data-quality.md](docs/data-quality.md).
- **AI label audit**: 마지막 라벨 감사의 결과(빠진 객체·느슨한 박스·클래스 뒤바뀜·근거 없는 박스)와 가장 의심스러운 이미지 목록. **Review … suspect images**는 Explore를 의심 점수 순으로 엽니다.
- **History**: 라벨 버전(스냅샷) 목록, 버전별 변경 내역(이미지별 박스 수, 클래스별 +/−), 전체 또는 이미지 단위 **Restore**(AI 제안 일괄 수락을 되돌리면 그 제안도 다시 대기 상태로), **Save version**. 보관 개수와 Restore·Undo의 차이는 아래 [라벨 파일 안전](#라벨-파일-안전).
- **Dataset**: 루트와 분할별 폴더, `data.yaml` 쓰기/갱신(주석 유지, `data.yaml.bak` 백업), 마지막 인덱싱 보고서(접힌 상태 — 문제 종류별 개수와 예시 최대 5개, 예: 이미지 폴더가 없는 라벨 폴더), **Dataset card**(`DATASET_CARD.md` 다운로드).
- **Classes**: 클래스 이름 편집(중복 검사, 저장 전 초안 유지), 클래스마다 **Examples**(Objects에서 그 클래스의 박스 전체 보기)와 **Card**(AI에 주는 프롬프트·정의·포함/제외 기준·예시 박스). 이름이 `class_0`처럼 비어 있는 데이터셋에는 **Suggest names with AI**가 강조됩니다.
- 분포 차트: 클래스 분포, 분할별 통계, 이미지당 박스 수, 박스 크기·종횡비, 박스 중심 히트맵, 해상도.

### Explore
- 툴바: 파일 이름 검색, 분할, **Classes**(Any of / Only), **Filters**(라벨 상태, 리뷰 상태, AI 상태, 박스 수 범위, 라벨 파일·데이터 품질·AI 감사 이슈), 정렬(Index, Name, Boxes, Resolution, Modified, Reviewed, Random + **Shuffle**, Most suspicious first(데이터 품질 플래그로 거를 때만), Most suspect (AI audit), Lowest AI confidence), 타일 크기, 박스 표시, 저장된 보기. 활성 필터는 모두 지울 수 있는 칩으로 보입니다.
- **기본 정렬은 필터를 따릅니다**: 데이터 품질 플래그(`issue=seq_gap` 등)로 거르면 *Most suspicious ↓*(그 플래그의 점수 순), AI 감사 소견(`issue=ai_*`, `ai=audit`)은 *Most suspect ↓*, 그 밖은 Index ↑. 필터를 바꾸면 기본 정렬도 따라 바뀌고, 직접 고른 정렬은 URL에 남아 새로고침과 Annotate 링크에서도 유지됩니다.
- **필터 옆 숫자는 다른 활성 필터를 모두 적용한 개수**입니다(일반적인 패싯 검색 방식). 예를 들어 `split=train`일 때 Classes의 숫자는 train 안의 개수입니다. 아직 이전 필터 기준인 숫자는 흐리게 보이고, 0인 선택지는 회색입니다.
- 가상화된 그리드(수만 장도 DOM 수백 개), 타일 배지(◌ n = 대기 중인 AI 제안, ? n = AI 감사 의심, 점 = 이슈/품질 플래그), 필터에서 벗어난 이미지는 사라지지 않고 *No longer matches*로 흐려집니다.
- 리뷰: `A`/`F`/`X`/`U`(승인·플래그·제외·해제), `⇧A`는 아직 리뷰하지 않은 나머지만 승인. 50장 넘는 일괄 리뷰는 정확한 개수로 확인하고, 플래그·제외된 이미지를 덮어쓰기 전에 묻습니다. `⌘Z`로 되돌립니다.
- 선택: 클릭·`S`·`⇧클릭` 범위, 타일 사이 빈 곳에서 드래그해 사각형 선택, `⌘A`로 조건에 맞는 전체. 일괄 작업: Approve/Flag/Exclude, AI 제안 수락·거절(수락은 Annotate `⇧Enter`와 같은 규칙: 같은 클래스 박스와 겹치는 제안은 *already labeled*로 건너뛰고, 다른 클래스 박스에 겹친 ⇄ 제안은 대기로 남김), **Annotate**, **⋯** 메뉴(선택만 보기, 경로 복사, 경로 목록 `.txt`, 이미지+라벨 ZIP).
- **작업 묶음(working set)**: 1,500장 이하의 선택은 `?ids=`로, 그보다 크거나 *all matching*인 선택은 서버에 순서째 저장된 묶음(`?set=<token>`, 최대 200,000장)으로 Annotate·Objects·AI 탭에 넘어갑니다. *Most suspicious* 정렬에서는 작은 선택도 묶음으로 넘겨 목록 순서를 그대로 지킵니다. 묶음은 마지막 사용 후 24시간 유지되고, 만료되면 *This selection is no longer available*과 함께 빠져나갈 길을 보여 줍니다.
- **Ask AI**: 검색창 왼쪽 ✦(또는 빈 검색창에서 `⇧/`)로 자연어 검색 — "넘어진 사람이 있는 val 이미지", "blurry frames with more than 3 boxes". 생성(generate)이 가능한 공급자를 이 브라우저에서 쓸 수 있을 때만 보입니다(다른 PC의 브라우저에는 클라우드 공급자가 빠짐 — [보안](#보안)의 원격 AI 차단). 질문과 데이터셋의 클래스·분할 정보만 전송되고 이미지는 보내지 않으며, 결과 필터 앞에 한 줄 설명·모델·비용이 붙고 **Undo**나 `Esc`로 원래 필터로 돌아갑니다. 같은 질문은 10분간 무료로 캐시됩니다.
- **Inspect**(Enter/클릭): 원본 이미지 확대·이동, 박스별 이슈, 리뷰와 메모(`N`), 대기 중인 AI 제안 수락(`⇧Enter`)·거절(`Delete`), 감사 수정 적용, **Ask AI to check the labels**(`I`), 비슷한 이미지(dHash) 12장, 시퀀스로 필터. `Space`를 누르고 있으면 그리드에서 바로 미리보기.

### Objects
박스 단위 갤러리. 모든 박스를 크롭 타일로 보여 주고 클래스·면적·종횡비·AI 감사 의심도 순으로 정렬합니다. 숫자 키로 클래스 재지정, `Del`로 삭제, `A`로 AI 감사 제안 수락, `Enter`로 해당 박스가 선택된 채 Annotate로 이동, `Space`를 누르고 있으면 전체 프레임. 수정은 스냅샷을 남기고 `⌘Z`로 되돌리며(수락한 AI 제안도 다시 대기 상태로), 다른 곳에서 바뀐 파일은 이미지별 충돌로 표시됩니다. 느슨한 박스 감사 결과는 제안 박스를 점선으로 보여 주고, `?set=`로 받은 선택은 *N selected images* 칩으로 표시됩니다.

### Annotate
- 도구: `V` 선택 · `B` 박스(`B`를 한 번 더: 드래그 ⇄ 2-클릭 — 첫 모서리 후 줌·이동 가능) · `P` 고정 크기 박스 · `W` 클릭으로 박스(SAM) · `H` 이동. 휠 줌, `Space` 드래그 이동, `0-9` 클래스, `[`/`]` 이전/다음 클래스, `/` 클래스 검색, `E`+`Alt+화살표`로 한 변만 이동, `O` 박스 숨기기, `M` 돋보기, `T` 십자선, `Z` 박스로 확대.
- 저장: 변경 0.5초 뒤 자동 저장, 이미지 이동 시 즉시 저장. 다른 곳에서 파일이 바뀌었으면 충돌 대화상자에서 **Merge both**(기본) · **Use the file on disk** · **Overwrite with mine** 중 고릅니다. 실행 취소는 이미지별로 유지됩니다.
- **AI 제안**: `I`로 현재 모델에 이 이미지의 제안 요청(기다리는 동안 다음 이미지로 넘어가도 결과는 요청한 이미지에 저장, 최대 3장 동시), `⇧I`로 모델 선택. 제안은 신뢰도와 함께 점선으로 그려지고, `Enter` 수락 · `⇧Enter` 보이는 것 모두 수락(다른 클래스 박스에 겹친 ⇄ 제안은 남김) · `Del` 거절 · `⇧Del` 모두 거절 · `{`/`}` 표시 임계값(기본 0.25, 패널의 `{` `}` 버튼도 같음). 제안을 끌거나 클래스를 바꾸면 그 자체가 수락입니다. 결정은 이미지를 떠날 때 서버에 저장되고, 닿지 않으면 저장 상태가 *Decisions not saved*로 바뀐 채 백그라운드에서 다시 보냅니다(`⌘S`는 바로). `⇧N`은 결정할 제안이 있는 다음 이미지(제안이 모두 임계값 아래이거나 이미 라벨된 박스와 겹쳐 숨은 이미지는 건너뜀), 숨은 제안은 패널의 **Reject hidden**으로 한 번에 거절합니다. AI 감사 의심(+ 빠짐, ⤢ 느슨함, ⇄ 클래스, ✕ 근거 없음)은 `Enter` 한 번으로 고칩니다.
- **시퀀스 도구**: 하단 **filmstrip**(`\`, 시퀀스의 모든 프레임 — 박스 수·제안·이슈·리뷰 표시, `,`/`.`로 프레임 이동), `` ` `` 이전 프레임 고스트, `C` 이전 프레임 박스를 제안으로(`⇧C`는 바로 복사), `K`로 키프레임 두 개 사이 **Interpolate**(같은 클래스 박스는 어디로 움직였든, 넘어짐처럼 제자리에서 클래스가 바뀐 박스도 짝지어 사이 프레임은 가까운 키프레임의 클래스를 받음), 한 프레임에만 빠진 박스(*Box missing in one frame of a sequence* 플래그)는 이미지 아래 플래그 칩 옆 **Fill from neighbours** 한 번으로 보간 제안(`Enter` 수락), *Propose boxes on the next new frame*, `R` 선택 박스를 SAM으로 다듬기. 데이터 품질 플래그 칩을 누르면 무엇이 다른지 그림 위에 주황색으로 보여 줍니다(문제의 박스는 테두리, 이웃 프레임·쌍둥이 이미지의 박스나 빠진 박스의 예상 위치는 점선. 칩을 다시 누르거나 `Esc`로 지움).
- 리뷰 `⇧A`/`⇧F`/`⇧X`(Advance after ⇧A / ⇧X 설정; 이미 같은 상태인 이미지에서 다시 누르면 해제하지 않고 그 설정대로 넘어감), `⇧U` 해제, **Range edit**(표시한 시작 이미지부터 현재까지 클래스 일괄 변경, Undo 가능), 작업 묶음 칩과 위치 입력(번호, `#id`, 파일 이름 일부). 링크에 정렬이 없으면 Explore와 같은 기본 순서로 엽니다(플래그 목록은 가장 의심스러운 것부터, 감사 목록은 의심 점수 순).

### AI 탭
**Overview**(사용액, 제안·의심 수, 공급자 상태) · **Label images**(일괄 라벨링, 대기 중인 제안 일괄 수락/거절) · **Two-stage**(검출기 → 분류기, 테스트 미리보기) · **Label audit** · **Benchmark**(내 라벨로 2–4개 설정 비교) · **Class cards**(**Suggest names with AI** 포함) · **Runs & cost**(실행 기록, 30일 사용액, *Provider batches in progress*). 일괄 라벨링의 기본 대상은 *Proposals*(라벨 파일을 건드리지 않음)이고, *Fill empty / Append / Replace*는 스냅샷 후 쓰며 실행 후 **Undo these changes**로 되돌립니다(그 뒤 직접 고친 파일은 그대로 둠). *Append*는 같은 클래스 박스와 겹치는 AI 박스를 중복으로 건너뛰고, 다른 클래스 박스에 겹치는 것은 쓰지 않고 제안(⇄)으로 남깁니다. 대기 중인 제안의 일괄 수락도 같은 규칙입니다. 배치 API가 있는 클라우드 공급자는 **Economy · −50%**(24시간 내 결과)를 고를 수 있습니다.

- **공급자**: 이 PC에서 도는 Ultralytics YOLO(닫힌 어휘), YOLOE(클래스 카드 텍스트 또는 내 라벨 예시로 찾는 열린 어휘), Segment Anything(박스 다듬기·클릭으로 박스)과, Anthropic Claude · Google Gemini · OpenAI · OpenAI 호환 서버(Ollama, LM Studio, vLLM). 모델 선택기는 *On this machine / Cloud / Pipelines*로 나뉘고 가격과 박스 정확도 메모를 함께 보여 줍니다.
- **키와 개인정보**: 키는 Settings → AI providers에 쓰기만 가능한 입력란으로 저장되고(또는 환경 변수), 화면에는 끝 4자리만 보입니다. 공급자마다 이미지가 어디로 가는지(*Images are sent to …* / *Images stay on this machine*) 표시하며, 로컬 모델은 이미지를 밖으로 보내지 않습니다.
- **비용**: 일괄 라벨링·라벨 감사·벤치마크·클래스 이름 제안은 시작 전에 예상 금액을 보여 주고(일괄 라벨링은 Now·Economy 둘 다), Two-stage 설정은 100장당 예상 금액을 보여 줍니다. 한 장짜리 호출 — Annotate `I` 제안, Inspect의 **Ask AI to check the labels**, Explore **Ask AI**, *Preview on 3 images*, Two-stage의 *Test on 4 labeled images* — 은 따로 추정하지 않고 바로 보냅니다. 이때는 모델 선택기의 1M 토큰당 가격이 기준이고, 결과 옆에 실제 비용이 붙습니다. 실행별 예산은 일괄 라벨링에만 있고, 공급자별 30일 지출 상한에 닿으면 새 실행과 한 장짜리 호출을 모두 거부합니다(일괄 라벨링·감사는 실행 도중에도 멈춤). 진행 중인 실행이 아직 과금되지 않은 몫(기다리는 Economy 배치, 남은 이미지)도 상한에서 미리 빼며, Settings에 *reserved by runs still going*으로 보입니다. 예산도 상한도 없는 클라우드 일괄 라벨링·감사는 예상 비용의 높은 쪽이 $5 이상이면 시작 전에 한 번 더 묻습니다(*Start a cloud run with no spend cap?*). 실제 토큰과 비용은 Runs & cost에 남습니다. 모델이 거절했거나 읽을 수 없거나 출력 한도에서 잘린 답도 공급자가 과금하므로 실패한 이미지의 비용까지 사용액·지출 상한에 들어갑니다.

공급자 선택, 비용 계산, 감사·2단계·벤치마크의 동작 방식과 문제 해결: [docs/ai-labeling.md](docs/ai-labeling.md).

### Compare (플러그인)
예측 라벨 폴더(6번째 열 = confidence) 또는 저장된 AI 실행(배치·벤치마크·**라벨 감사** 포함)을 GT와 비교합니다. COCO / Ultralytics / VOC 지표, 임계값과 무관한 mAP@.5·mAP@.5:.95, 신뢰도에 따른 F1·정밀도·재현율 곡선과 최적 F1 임계값, 혼동 행렬, 객체 크기별 AP, 이미지별 오류 표와 나란히 보기(`←`/`→`, `[`/`]` 임계값, `F` 플래그), FN/FP 이미지를 한 번에 리뷰 *flagged*로 표시하고 되돌리기, CSV 내보내기. **Run a model…**로 로컬 모델을 바로 돌려 평가할 수 있습니다(8프레임씩 배치 추론). 기하 모드 감사는 *person (0+1)*처럼 클래스 묶음 단위로 채점하고, *More options*의 **Count as one class**로 직접 묶을 수도 있습니다. 링크: `/d/<id>/compare?run=<run_id>&run_b=<id>`.

### Export
- **Subset**: 분할·클래스·라벨 상태·리뷰 상태로 고른 이미지를 **Download ZIP** 또는 **Copy to folder…**(리사이즈 옵션, `data.yaml` 포함). **Add a dataset card (DATASET_CARD.md)**를 켜면 분할·클래스(카드 정의 포함)·데이터 품질·리뷰 진행률·라벨 이슈·사용한 내보내기 설정을 담은 카드가 `data.yaml` 옆에 들어갑니다. 미리보기에서 내용을 보고 **Download this card**로 따로 받을 수 있습니다.
- **Clean-up & splits**(데이터셋은 바뀌지 않음): **Drop near-duplicate images**(근중복 그룹마다 승인 > 라벨 있음 > 박스 많음 > 먼저 인덱싱된 것(이미지 ID가 작은 것) 순으로 1장만), **Make new train / val / test splits**(비율·시드, *Frames of one sequence* 등 묶음 단위로 한 분할에만, 클래스 비율 균형), 클래스 유지·합치기·제외와 번호 압축, 클래스 목록 밖 ID 처리.
- 내보낸 `data.yaml`은 어느 작업 폴더에서 학습을 시작해도 동작하며(ZIP은 `path` 없이, 폴더 복사는 절대 경로), train/val이 항상 들어갑니다. 세그멘테이션·포즈 행은 좌표 그대로, confidence 열은 제거됩니다. 폴더 복사는 비어 있지 않은 폴더에 쓰기 전에 묻고(이전 Fovea 내보내기만 교체), 모든 내보내기에 `fovea-export.json`(필터·옵션·건너뛴 이미지)을 남깁니다. 리사이즈 없는 ZIP은 정확한 크기를 알려 브라우저가 진행률을 표시합니다.
- **Review lists**(승인·플래그·제외·학습 목록 `.txt`), **data.yaml** 쓰기. 카드만 필요하면 `GET /api/datasets/<id>/card.md`(필터를 쿼리로 붙이면 그 선택만, `inline=1`이면 브라우저에서 보기).

### Files · Settings
- **Files**: 데이터셋 루트 탐색(행을 클릭하거나 경로 줄의 **Copy path**에서 `Tab`으로 목록에 들어간 뒤 `↑`/`↓`/`Enter`로 이동, `Backspace`로 상위), 텍스트·이미지 미리보기, YOLO 라벨 파일은 클래스 칩과 박스를 이미지 위에 그려 줍니다.
- **Settings**: *Appearance*(테마), *Visual comfort*(화면 어둡게, 이미지 주변 배경, 색각 이상 안전 클래스 색, 휴식 알림 — 기기별), *AI providers*(상태, 키 입력·교체·삭제·Test, 기본 모델, 노력 수준, 30일 지출 상한, OpenAI *Flex tier*, 병렬 요청, OpenAI 호환 서버 URL과 좌표 형식, 로컬 추론 장치), *Model weights*(정해진 목록에서 명시적 Download), *Storage & server*(썸네일·크롭 캐시 통계와 **Clear caches**, 서버 접속 범위).

### 어디서나
`⌘K`(Windows/Linux `Ctrl+K`) 명령 팔레트 — 데이터셋·페이지·이미지·동작 검색. `?` 현재 화면의 단축키. `G` 다음 `H/O/E/B/A/I/F/C/X/S`로 Home·Overview·Explore·Objects·Annotate·AI·Files·Compare·Export·Settings 이동. 실행 중인 잡은 데이터셋 헤더의 고정 높이 상태 줄과 사이드바 *Activity*에 경과 시간·취소 버튼과 함께 보입니다. 단축키 전체: [docs/keyboard.md](docs/keyboard.md). 한글 입력 상태나 Caps Lock에서도 물리 키 기준으로 동작합니다.

## 라벨 파일 안전

- Fovea는 이미지 파일을 쓰거나 지우지 않고, 라벨 파일은 편집한 것만 원자적으로(임시 파일 → rename) 다시 씁니다. 저장은 파일 버전을 확인해 다른 편집을 덮어쓰지 않습니다.
- 일괄 라벨 작업(클래스 변경·리맵·삭제·비우기, Range edit, AI 일괄 쓰기, 제안 일괄 수락, Objects 편집) 직전에는 바뀔 라벨 파일의 **스냅샷**을 자동으로 남기고, Overview의 History에서 차이를 보고 되돌릴 수 있습니다. 작업이 끝나면 스냅샷을 그 작업이 실제로 바꾼 파일로 줄이고, 아무것도 바꾸지 않았으면 되돌리기 지점을 남기지 않습니다(결과가 더 들어올 수 있는 Economy 실행은 줄이지 않음). 되돌리기 직전에도 스냅샷을 남기므로 되돌리기 자체도 취소할 수 있습니다. AI 제안 일괄 수락의 스냅샷은 그 작업이 수락한 제안도 기억하므로, 되돌리면 그 제안도 다시 대기 상태가 됩니다(*N AI proposals pending again*. 되돌리지 않고 남긴 파일의 이미지는 제외).
- **스냅샷 보관**: 자동 스냅샷은 데이터셋마다 두 묶음으로 따로 보관합니다 — 파일이 20개를 넘는 것은 최근 20개, 20개 이하인 작은 것(Objects 키 한 번, 이미지 몇 장 편집)은 최근 50개. 그래서 작은 편집이 수십 번 이어져도 일괄 작업이나 AI 쓰기의 되돌리기 지점이 밀려나지 않습니다. 직접 저장한 버전(**Save version**)은 지울 때까지 남습니다. 오래된 스냅샷 정리는 백그라운드에서 하므로 편집이 기다리지 않습니다.
- **작업의 Undo와 History의 Restore는 다릅니다.** 작업 자체의 Undo — AI 일괄 실행의 **Undo these changes**, Annotate **Range edit**의 **Undo**, Objects의 `⌘Z` — 는 그 작업이 쓴 파일만, 그 파일이 작업이 남긴 내용 그대로일 때만 되돌립니다. 그 뒤 Annotate나 다른 도구에서 고친 파일은 덮어쓰거나 지우지 않고 그대로 두며 *N edited since — kept*로 알려 줍니다(예: AI *Fill empty*가 건너뛴 이미지에 나중에 직접 만든 라벨은 Undo해도 남음). History의 **Restore**는 그 버전 전체를 되돌립니다: 이후의 편집도 버전 내용으로 바뀌므로, 확인 창이 바뀔 파일 수를 먼저 보여 줍니다.
- 되돌리는 동안 다른 곳(Annotate, 다른 탭, 외부 도구)에서 저장된 파일은 덮어쓰지 않습니다. 그 새 편집을 남기고 결과에 *N changed meanwhile and kept*로 알립니다(Undo와 Restore 모두).
- **심볼릭 링크인 라벨 파일**: 링크가 데이터셋 폴더(루트나 분할의 이미지·라벨·목록 폴더) 안의 파일을 가리키면 링크는 그대로 두고 그 파일에 씁니다(링크를 일반 파일로 바꾸면 공유하던 원본이 낡은 채 남으므로). 데이터셋 밖을 가리키면 쓰지 않습니다: 편집기 저장은 이유와 함께 실패하고(API는 403 `label_link_outside`), 일괄 작업은 그 파일을 건너뛰고 몇 개를 건너뛰었는지 알려 주며, Objects는 그 이미지를 충돌로 표시합니다. 가리키는 파일을 직접 고치거나 링크를 일반 파일로 바꾸세요. 이미지 자리의 링크가 밖을 가리키는 경우는 인덱싱 보고서의 `link_outside`입니다([docs/data-quality.md](docs/data-quality.md#인덱싱-보고서)).
- **Fovea가 라벨 파일을 지우는 경우는 두 가지입니다.** (1) 되돌리기(History의 **Restore**, 작업의 Undo)가 스냅샷 시점에 없던 라벨 파일을 지웁니다(예: AI *Fill empty*가 새로 만든 파일은 Undo하면 지워짐. 되돌리기 직전 스냅샷이 남으므로 취소할 수 있음). (2) Annotate가 박스 없는 라벨 파일을 지웁니다: 라벨 파일이 없던 이미지(또는 이번 세션에 파일을 만든 이미지)의 박스가 다시 0개가 될 때(그렸다가 실행 취소하거나 모두 삭제)와, 빈 라벨 파일이 있는 이미지에서 **Remove file**을 누를 때(일부러 만든 배경 프레임도 지워짐)입니다. (2)는 스냅샷을 남기지 않습니다. 원래 라벨 파일이 있던 이미지는 박스를 모두 지워도 빈 파일(배경 프레임)로 남고, **Mark as background**는 빈 파일을 일부러 만듭니다.
- 세그멘테이션·포즈·OBB 행은 박스로 표시되지만 원래 좌표 그대로 보존되며, 클래스 변경·삭제만 가능합니다(이동·크기 변경은 거부).
- 라벨이 저장될 때마다(편집기, 일괄 작업, Objects, 되돌리기, AI 쓰기) 그 이미지의 시퀀스·근중복 쌍둥이·박스 모양 플래그를 약 3초 뒤 다시 검사하므로 데이터 품질 결과가 편집을 따라옵니다.

## 시각 피로

몇 시간씩 이미지를 넘겨 보는 작업을 전제로 설계했습니다. 이미지 전환은 항상 즉시(다음 이미지가 디코딩될 때까지 현재 이미지를 유지한 뒤 한 번에 교체) 일어나며, 페이드·빈 화면·깜박이는 자리표시자·반복 애니메이션·클릭 대상이 밀리는 레이아웃 변화가 없습니다. AI 제안과 고스트는 정적인 점선, 잡 진행은 고정 높이 상태 줄에 표시됩니다. 규칙과 2024–2026 연구 근거: [docs/visual-comfort.md](docs/visual-comfort.md).

## 보안

Fovea에는 사용자 계정이 없습니다. 그래서 경계는 *누가 서버에 닿을 수 있는가*입니다.

- **기본은 이 PC만**: `127.0.0.1`에서 듣고, Docker도 `127.0.0.1:8888`로만 엽니다.
- **Host 검사**(DNS 리바인딩 차단): localhost·IP 주소·이 PC 이름·`FOVEA_ALLOWED_HOSTS`만 통과합니다(그 외 400).
- **Origin 검사**(CSRF 차단): 다른 사이트의 페이지가 보낸 POST/PUT/PATCH/DELETE는 403. 역방향 프록시가 Host를 바꾸면 `FOVEA_ALLOWED_ORIGINS`에 공개 주소를 넣으세요(아래 *역방향 프록시*). 다른 사이트가 Fovea를 iframe에 넣을 수 없습니다.
- **접속 토큰**: `FOVEA_TOKEN`을 설정하면 다른 PC는 토큰 입력 페이지(또는 `Authorization: Bearer`)를 거쳐야 합니다. 이 PC에서는 묻지 않습니다. "이 PC"는 접속한 주소로 판단합니다: 루프백(`127.0.0.1`, `::1`)이나 `FOVEA_LOCAL_NETS`에 든 주소만 이 PC입니다. 그래서 Docker를 다른 PC에 열면(`FOVEA_LOCAL_NETS`를 지우면) Docker를 실행한 PC도 토큰을 묻습니다(위 *Docker*). 역방향 프록시 뒤라면 아래 항목을 꼭 보세요.
- **원격 AI 차단**: 클라우드 공급자는 소유자의 돈을 씁니다. 다른 PC에서 온 요청은 키를 저장·삭제하거나 클라우드 모델을 호출할 수 없고, 저장된 키를 보내는 OpenAI 호환 서버(유료 게이트웨이일 수 있음)도 호출할 수 없습니다(`FOVEA_ALLOW_REMOTE_AI=1` 또는 `--allow-remote-ai`로 해제). 브라우저가 다른 PC인지는 주소창이 아니라 서버가 판단하고(`/api/meta`의 `request_is_local`), 그런 브라우저에서는 AI 탭에 *Cloud AI off from this browser*가 뜨고 막힌 공급자가 비활성으로 보입니다. 다른 PC에 연 Docker에서는 Docker를 실행한 PC도 여기에 걸리므로, 클라우드 AI를 쓰려면 `.env`에 `FOVEA_ALLOW_REMOTE_AI=1`을 넣어야 합니다.
- **역방향 프록시**(nginx·Caddy로 공개 주소나 HTTPS를 붙일 때): 같은 PC의 프록시가 원래 클라이언트 주소를 `X-Forwarded-For`로 넘기지 않으면 모든 접속이 프록시 주소(`127.0.0.1`), 즉 **이 PC로 취급되어 `FOVEA_TOKEN`과 원격 AI 차단이 적용되지 않습니다.**
  - nginx: `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`(또는 `$remote_addr`)를 반드시 넣으세요. 이 줄이 없으면 클라이언트가 직접 보낸 `X-Forwarded-For: 127.0.0.1`이 그대로 전달되어 이 PC로 위장할 수 있습니다. Caddy의 `reverse_proxy`는 기본으로 넣습니다.
  - 서버(uvicorn)는 이 PC의 루프백에서 온 `X-Forwarded-For`만 믿습니다(기본 `127.0.0.1`, uvicorn 0.53부터는 `127.0.0.1,::1`). 프록시가 다른 PC에 있으면 `FORWARDED_ALLOW_IPS=<프록시 주소>`를 설정하세요. `*`는 클라이언트가 쓴 첫 값을 믿게 되므로 쓰지 마세요. Docker에서는 호스트의 프록시가 컨테이너에 Docker 브리지 주소로 보이므로 `docker-compose.yml`의 `environment`에 `FORWARDED_ALLOW_IPS=172.16.0.0/12,192.168.65.0/24`(`FOVEA_LOCAL_NETS`와 같은 범위)를 추가하세요.
  - 프록시가 `Host`를 `127.0.0.1:8000` 등으로 바꾸면 `FOVEA_ALLOWED_ORIGINS`에 공개 주소(`https://fovea.example.com`)를, 공개 이름을 그대로 넘기면(`proxy_set_header Host $host`) `FOVEA_ALLOWED_HOSTS`에 그 이름을 넣으세요. Docker에서는 두 변수 모두 `.env`에 넣으면 전달됩니다.
- **API 키**: 환경 변수가 우선이고, Settings에서 입력한 키는 `<data-dir>/secrets.json`(권한 0600)에 저장됩니다. 어떤 응답에도 키가 나오지 않고(있음 여부와 끝 4자리만), 오류 메시지·로그·잡 결과에서는 키 모양 문자열을 지웁니다. 일반 설정 저장소는 key/secret/token 이름의 값을 거부합니다.
- 너무 큰 이미지는 디코딩하지 않습니다(`FOVEA_MAX_IMAGE_PIXELS`, 기본 2.5억 픽셀).

### 환경 변수

| 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `FOVEA_DATA_DIR` / `FOVEA_MODEL_DIR` | `src/data` / `src/model` | `--data-dir` / `--model-dir`와 같음 |
| `FOVEA_TOKEN` | — | 다른 PC에 요구할 접속 토큰 |
| `FOVEA_ALLOWED_HOSTS` | — | 추가 허용 호스트 이름(쉼표, `*.example` 가능, `*`는 검사 끔) |
| `FOVEA_ALLOWED_ORIGINS` | — | 역방향 프록시 뒤의 공개 Origin |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` (uvicorn 0.53+는 `127.0.0.1,::1`) | `X-Forwarded-For`를 믿을 프록시 주소(uvicorn 설정, 쉼표·CIDR 가능, `*` 금지) |
| `FOVEA_LOCAL_NETS` | — | "이 PC"로 취급할 클라이언트 네트워크(Docker 브리지) |
| `FOVEA_ALLOW_REMOTE_AI` | 꺼짐 | 다른 PC의 클라우드 AI 사용 허용 |
| `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`(또는 `GOOGLE_API_KEY`), `OPENAI_API_KEY`, `OPENAI_COMPAT_API_KEY` | — | 공급자 키(Settings에 저장한 키보다 우선) |
| `FOVEA_ANTHROPIC_BASE_URL`, `FOVEA_GEMINI_BASE_URL`, `FOVEA_OPENAI_BASE_URL`, `FOVEA_OPENAI_COMPAT_BASE_URL` | 공식 주소 | 게이트웨이·프록시 주소(설정에 저장한 주소 — OpenAI 호환 서버의 *Server URL* — 가 있으면 그것이 우선). `https://user:password@gateway/v1`처럼 자격 증명을 넣으면 요청에 basic auth로 붙이고, 화면·API 응답·로그·오류 메시지·저장된 배치 정보에는 호스트만 나옵니다. Settings의 *Server URL*은 자격 증명이 든 주소를 거부합니다(키 칸에 넣기) |
| `FOVEA_THUMB_CACHE_MAX_GB` | 10 | 썸네일 캐시 상한(넘으면 오래 안 쓴 것부터 삭제, 0 = 무제한) |
| `FOVEA_CROP_CACHE_MB` | 2048 | Objects 크롭 캐시 상한 |
| `FOVEA_MAX_IMAGE_PIXELS` | 250,000,000 | 디코딩을 거부할 픽셀 수 |
| `FOVEA_AI_BATCH_POLL_S` | 60 | Economy 배치 상태 확인 주기(초) |
| `FOVEA_AI_MOCK` | 꺼짐 | 테스트·UI 개발용 Mock 공급자 표시 |

## 구조

```
src/
├─ fovea/                   # Python 패키지 (FastAPI)
│  ├─ __main__.py           # python -m fovea (CLI 옵션, 깔끔한 종료)
│  ├─ main.py               # 앱 조립: 보안 미들웨어, 라우터, SPA 셸, 플러그인 마운트
│  ├─ security.py           # 로컬 요청 판별, Host·Origin 검사, 토큰, 원격 AI 차단, 비밀값 지우기
│  ├─ config.py paths.py    # 설정, 호스트↔컨테이너 경로 매핑
│  ├─ db.py jobs.py         # SQLite(WAL) 스키마·마이그레이션, 백그라운드 잡 + SSE
│  ├─ core/                 # layout(레이아웃 감지) · scanner(인덱싱) · labels(파싱·쓰기·쓰기 훅) · imagesq(필터·정렬·작업 묶음·패싯)
│  │                        # stats · thumbs · export(+DATASET_CARD.md) · qa(데이터 품질) · snapshots(라벨 버전) · yolo(구 API)
│  ├─ ai/                   # registry · secrets · config · pipeline · runner(배치·감사·벤치마크) · economy(공급자 배치 API)
│  │  │                     # assist(클래스 이름 제안·자연어 검색) · estimate · geometry · local · vlm · weights · prices.json
│  │  └─ providers/         # yolo · yoloe · sam · anthropic · gemini · openai · openai_compat · mock
│  ├─ api/                  # REST: datasets · images · objects · qa · snapshots · ai · export · fs · jobs · meta
│  ├─ plugin_api.py         # Plugin 매니페스트 + 자동 탐색
│  └─ plugins/compare/      # Compare 플러그인 (evaluate · analysis · store + static/)
├─ static/                  # 빌드 없는 ES 모듈 프런트엔드
│  ├─ app.js                # 셸·라우터·사이드바·명령 팔레트·전역 단축키·플러그인 로딩
│  ├─ fovea.css  css/       # 디자인 시스템(라이트/다크) + 화면별 스타일(annotate·explore·objects·ai)
│  ├─ lib/                  # dom · router · store · api · ui · keys · cursor(작업 묶음 커서) · media(디코딩 후 교체) · overlay · colors · comfort · icons
│  └─ views/                # home · dataset · overview · explore(+Inspect) · objects · annotate · ai · ai-panel · files · export · settings · issues
├─ templates/app.html       # SPA 셸 (import map)
├─ tests/                   # pytest + fake_vlm.py(클라우드 API 가짜 서버) + make_synth.py(합성 데이터셋)
├─ data/                    # fovea.db, thumbs/, crops/, secrets.json … (gitignored)
└─ model/                   # 모델 가중치 (gitignored)
```

## 플러그인 만들기

`src/fovea/plugins/<id>/__init__.py` 에 `PLUGIN` 을 노출하면 자동으로 로드됩니다.

```python
from pathlib import Path
from fastapi import APIRouter
from fovea.plugin_api import Plugin

router = APIRouter()

@router.get("/hello")
def hello(dataset_id: str):
    return {"hi": dataset_id}

PLUGIN = Plugin(
    id="hello", name="Hello", description="example",
    router=router,                                   # → /api/plugins/hello/...
    static_dir=Path(__file__).parent / "static",     # → /plugins/hello/...
    entry="hello.js",                                # 프런트 진입 모듈
    nav=[{"id": "hello", "label": "Hello", "icon": "puzzle", "order": 60}],
)
```

```js
// static/hello.js — default export receives the app API
export default function install(fovea, manifest) {
  fovea.registerTab({ id: 'hello', label: 'Hello', icon: 'puzzle', order: 60,
    render: async (el, { dataset }) => {
      const r = await fovea.api.get(`${manifest.api}/hello`, { dataset_id: dataset.id });
      el.appendChild(fovea.h('div', { class: 'page-inner' }, `hi ${r.hi}`));
    } });
  fovea.registerCommand({ id: 'hello.say', label: 'Say hello', icon: 'zap', run: (f) => f.ui.toast('hello') });
}
```

`fovea` 객체가 제공하는 것: `h/svg/icon/fmt/cls` DOM 헬퍼, `api.get/post/put/patch/del/watchJob/imgUrl/thumbUrl`, `router`, `link`, `ui`(toast/modal/confirm/menu/seg/drawer/spinner/elapsedClock/…), `colors`(`classColor`, `textOn`), `boxLayer`, `ImageCursor`, `media`(`createFrameCache`·`trackImage` — 디코딩 후 한 번에 교체), `keys`(`is`로 물리 키 매칭, `registerKeys(scope, title, [[키, 설명]…])`로 `?` 도움말에 등록), `comfort`, `state`, `bus`, `jobs`, `pickPath`(파일 브라우저), `registerTab`, `registerCommand`. 새 화면은 [docs/visual-comfort.md](docs/visual-comfort.md)의 체크리스트를 따릅니다.

## 테스트

```bash
cd src
pip install -r requirements-dev.txt
python -m pytest tests -q
```

테스트는 기본적으로 `src/tests/.tmp-data`(gitignored)를 데이터 폴더로 쓰므로 실제 인덱스를 건드리지 않습니다(`FOVEA_DATA_DIR`로 바꿀 수 있음). 프런트엔드 테스트 일부는 Node.js가 있을 때만 실행되고, 클라우드 공급자 테스트는 `tests/fake_vlm.py`의 가짜 서버를 쓰므로 키나 네트워크가 필요 없습니다.

## 문서

- [docs/ai-labeling.md](docs/ai-labeling.md) — AI 공급자, 제안 검토, 일괄·Economy, 2단계, 라벨 감사, 벤치마크, 비용, 키와 개인정보, 문제 해결
- [docs/data-quality.md](docs/data-quality.md) — 데이터 품질 플래그의 의미·계산·임계값·오탐
- [docs/keyboard.md](docs/keyboard.md) — 화면별 단축키
- [docs/visual-comfort.md](docs/visual-comfort.md) — 시각 피로 최소화 원칙과 근거
