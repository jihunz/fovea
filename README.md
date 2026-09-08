<p align="center">
  <img src="docs/logo.svg" width="72" alt="Fovea" />
</p>
<h1 align="center">Fovea</h1>
<p align="center"><b>Local-first computer-vision dataset review &amp; labeling workspace.</b><br/>
YOLO 포맷 데이터셋을 등록하면 인덱싱 → 헬스체크 → 갤러리 검수 → 라벨링 → 내보내기까지 한 화면에서. 업로드도, 계정도 없습니다.</p>

---

> 이 저장소는 `cv-data-viewer`를 **Fovea**로 전면 재설계한 버전입니다. 저장소 이름도 `fovea`로 바꾸는 것을 권장합니다
> (`gh repo rename fovea` 또는 GitHub 설정에서 변경 후 `git remote set-url origin …`).

## 왜 Fovea인가

망막에서 시력이 가장 예리한 지점이 **fovea(중심와)** 입니다. Fovea는 데이터셋을 가장 자세히 들여다보는 도구를 지향합니다.

| 이전 (cv-data-viewer) | Fovea |
| --- | --- |
| 매번 경로를 입력해 4분할 뷰어를 여는 무상태 도구 | 데이터셋을 **한 번 등록**하면 SQLite 인덱스(이미지·라벨·박스·해상도)가 만들어지고 언제든 다시 열림 |
| HTML에 전체 이미지 목록을 내려보냄 (수만 장에서 느림) | 서버 페이지네이션 + 썸네일 캐시 → **45k 이미지 인덱싱 17초**, 갤러리 즉시 응답 |
| 제외 목록만 브라우저 메모리에 저장 | **승인 / 플래그 / 제외** 리뷰 상태가 영구 저장, 진행률 표시, 목록·서브셋 내보내기 |
| 통계 없음 | **Overview 헬스체크**: 클래스 분포, 분할별 통계, 박스 크기·위치 히트맵, 라벨 이슈 자동 탐지 |
| 줌 없는 캔버스 어노테이터 | 줌/팬, 8방향 리사이즈, 클래스 팔레트·단축키, Undo/Redo, **자동 저장**, 범위 일괄 편집, AI 자동 라벨(서버 잡) |
| Compare 모드 하드코딩 | **플러그인**으로 분리 (`fovea/plugins/compare`) — P/R/F1·AP@IoU·이미지별 오류·나란히 보기 |

## 빠른 시작

### 로컬 (Python 3.11+)

```bash
cd src
pip install -r requirements.txt        # ultralytics/torch는 선택 — AI 자동 라벨에만 필요
python -m fovea --port 8000            # http://localhost:8000
```

### Docker

```bash
cd src
cp .env.example .env                   # HOST_PATH = 데이터셋이 있는 폴더 (컨테이너의 /host 로 마운트)
docker compose up --build -d           # http://localhost:8888
```

호스트 경로(`/Users/me/data/...`, `C:\data\...`)를 그대로 입력하면 컨테이너 경로로 자동 매핑됩니다. 인덱스 DB와 썸네일 캐시는 `src/data/`에, YOLO 가중치는 `src/model/`에 둡니다.

## 사용 방법

1. **Add dataset** — 폴더, `data.yaml`, `train.txt` 중 아무거나 지정하면 레이아웃을 자동 감지합니다.
   지원 레이아웃: `images/<split>+labels/<split>`, `<split>/images+labels`, `images/+labels/`, 이미지·txt 혼합 폴더, 라벨 없는 폴더(라벨 폴더 자동 생성), data.yaml, train.txt.
2. **Overview** — 이미지/라벨/박스/클래스 수, 리뷰 진행률, 헬스체크(라벨 누락·빈 라벨·범위 밖 박스·클래스 ID 오류·중복·미세 박스), 클래스 분포, 박스 크기·종횡비·중심 히트맵, 해상도 분포, 클래스 이름 편집, `data.yaml` 쓰기. 이슈 항목의 **Review** 를 누르면 해당 이미지만 필터된 갤러리로 이동합니다.
3. **Explore** — 검색·분할·클래스·라벨 상태·리뷰 상태·이슈·정렬 필터, 박스 오버레이, 타일 크기, 무한 스크롤. 타일 클릭 → **Inspect**(원본 + 라벨 목록 + 메타 + 리뷰). `⇧클릭`/`S` 다중 선택 후 일괄 승인/플래그/제외/어노테이트/경로 복사.
4. **Annotate** — `V` 선택 · `B` 박스 · `P` 포인트(고정 크기 박스) · `H` 팬, 휠 줌, `0-9` 클래스, `Del`, `⌘Z/⌘⇧Z`, `Tab`, `f` 화면 맞춤, `N` 다음 미라벨 이미지, `C` 이전 이미지 박스 복사, `⇧A/⇧F/⇧X` 리뷰. 변경은 0.5초 뒤 자동 저장되며 이동 시 즉시 저장됩니다.
   * **Propagate** 옵션: 라벨이 없는 다음 프레임에 현재 박스를 그대로 넘깁니다(연속 프레임용).
   * **Range edit**: 시작 지점을 표시한 뒤 현재 이미지까지의 모든 박스 클래스를 한 번에 변경.
   * **AI auto-label**: `src/model/*.pt`의 YOLO 모델로 현재 이미지 또는 N장/필터 전체를 서버 잡으로 라벨링. 모델 클래스 → 데이터셋 클래스 매핑, 모드(Fill/Append/Replace) 지원.
5. **Files** — 데이터셋 루트 탐색, 텍스트/이미지 미리보기.
6. **Export** — 필터(분할·클래스·라벨 상태·리뷰 제외)로 서브셋을 **ZIP 다운로드** 또는 **서버 측 폴더 복사**(리사이즈 옵션, `data.yaml` 포함), 리뷰 목록/학습 목록 `.txt`, `data.yaml` 생성.
7. **Compare(플러그인)** — 예측 라벨 폴더(6번째 열 = confidence)를 GT와 비교. 모델 A/B, IoU·conf 임계값, 클래스별 P/R/F1/AP, 최악 이미지 순 정렬, 나란히 보기(GT/FN/TP/FP 색상), PNG 클립보드 복사.

`⌘K` 명령 팔레트, `?` 단축키 도움말, `G` 다음 `H/O/E/A/F` 로 페이지 이동.

## 구조

```
src/
├─ fovea/                 # Python 패키지
│  ├─ main.py             # FastAPI 앱, SPA 셸, 플러그인 마운트
│  ├─ config.py paths.py  # 설정, 호스트↔컨테이너 경로 매핑
│  ├─ db.py jobs.py       # SQLite(WAL) 저장소, 백그라운드 잡 + SSE
│  ├─ core/               # layout(레이아웃 감지) · scanner(인덱싱) · labels · stats · thumbs · yolo · export · imagesq(필터 쿼리)
│  ├─ api/                # REST: /api/datasets /api/img /api/thumb /api/ai /api/jobs /api/fs /api/meta
│  ├─ plugin_api.py       # Plugin 매니페스트 + 자동 탐색
│  └─ plugins/compare/    # 첫 플러그인 (라우터 + static/compare.js)
├─ static/                # 빌드 없는 ES 모듈 프런트엔드
│  ├─ fovea.css           # 디자인 시스템 (라이트/다크)
│  ├─ app.js              # 셸·라우터·사이드바·커맨드 팔레트·플러그인 로딩
│  ├─ lib/                # dom · router · api · ui(모달/메뉴/토스트) · cursor(페이지 커서) · overlay · icons
│  └─ views/              # home · dataset(셸) · overview · explore(+inspect) · annotate · files · export · settings
├─ templates/app.html     # SPA 셸
├─ tests/                 # pytest (레이아웃 감지, 라벨 파싱, 매칭/AP)
└─ data/                  # fovea.db + thumbs/ (gitignored)
```

API 문서: `http://localhost:8000/api/docs`

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

`fovea` 객체가 제공하는 것: `h/svg/icon` DOM 헬퍼, `api.get/post/put/patch/del/watchJob`, `router`, `ui`(toast/modal/confirm/menu/seg/…), `colors`, `boxLayer`, `ImageCursor`, `state`, `bus`, `pickPath`(파일 브라우저), `registerTab`, `registerCommand`.

플러그인 후보: CSV 기반 클래스 자동 지정(구 어노테이터의 CSV Class Map), 시퀀스 단위 리뷰, 중복 이미지 탐지, 모델 학습 런 추적.

## 테스트

```bash
cd src && python -m pytest tests -q
```

## 참고

* 루트의 `subset_class1.py`(클래스 1 포함 이미지 복사)는 **Export → Subset(Classes 필터, Copy to folder)** 로 대체됩니다.
* 라벨 파일은 항상 원자적으로(임시 파일 → rename) 기록되며, Fovea가 스스로 파일을 삭제하는 경우는 없습니다.
