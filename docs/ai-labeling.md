# AI 라벨링

Fovea의 AI는 사람의 라벨링을 돕는 도구다. 원칙은 네 가지다.

1. **AI 결과는 기본적으로 제안(proposal)이다.** 점선 박스로 저장되고, 사람이 수락해야 라벨 파일에 들어간다. 라벨 파일에 바로 쓰는 것은 일괄 작업에서 쓰기 대상을 직접 고를 때뿐이며, 그때도 먼저 스냅샷을 남긴다.
2. **큰 비용은 실행 전에 보이고, 모든 호출에 상한이 걸린다.** 일괄 라벨링·2단계 설정·라벨 감사·벤치마크·클래스 이름 제안은 시작 전에 예상 금액을 보여 주고, 일괄 라벨링은 실행별 예산으로도 멈춘다. 한 장짜리 호출(Annotate `I` 제안, Inspect의 라벨 검사, Explore Ask AI, 미리보기)은 따로 추정하지 않고 모델의 토큰 가격만 보인다. 공급자별 30일 지출 상한(Settings)에 닿으면 모든 클라우드 호출이 거부되고, 예산도 상한도 없는 큰 실행은 시작 전에 한 번 더 묻는다(9절).
3. **키는 서버 밖으로 나가지 않는다.** 브라우저·URL·로그·오류 메시지·잡 결과 어디에도 키가 나오지 않는다.
4. **어떤 모델이 좋은지는 내 데이터에서 잰다.** 공개 벤치마크 순위는 도메인마다 뒤집히므로, Benchmark와 Compare로 내 라벨에 대고 비교한다.

## 1. 공급자

| 공급자 | 어디서 실행 | 무엇에 쓰나 | 필요한 것 |
| --- | --- | --- | --- |
| **Ultralytics YOLO** (`yolo`) | 이 PC | 닫힌 어휘 검출(COCO 80 클래스 등). 2단계의 검출기, 감사 | `ultralytics`, `torch`, `src/model`의 `.pt` |
| **YOLOE** (`yoloe`) | 이 PC | 열린 어휘 검출: 클래스 카드의 텍스트 프롬프트(*Text*) 또는 내 라벨에서 만든 예시 프로토타입(*Your labels*) | 위와 같음 + `yoloe-26*-seg.pt`, 텍스트 모드는 `mobileclip2_b.ts` |
| **Segment Anything** (`sam`) | 이 PC | 박스 다듬기(`R`), 클릭으로 박스(`W`) | `sam2.1_t.pt`(기본) 또는 `mobile_sam.pt` |
| **Anthropic Claude** (`anthropic`) | api.anthropic.com | VLM 검출·분류, 클래스 이름 제안, 자연어 검색, Economy | API 키, `anthropic` 패키지 |
| **Google Gemini** (`gemini`) | generativelanguage.googleapis.com | 위와 같음 | API 키, `httpx` |
| **OpenAI** (`openai`) | api.openai.com | 위와 같음 | API 키, `httpx` |
| **OpenAI-compatible server** (`openai_compat`) | 설정한 주소(기본 `localhost:11434`, Ollama) | Ollama·LM Studio·vLLM·게이트웨이의 비전 모델. 비용 0 | 실행 중인 서버, `httpx` |
| Mock (`mock`) | 이 PC | 테스트·UI 개발용. `FOVEA_AI_MOCK=1`일 때만 보임 | — |

Settings → *AI providers*에서 상태(사용 가능 여부와 이유), 키, 기본 모델, 노력 수준, 지출 상한, 병렬 요청 수(OpenAI는 *Flex tier*도)를 관리한다. 각 공급자 옆의 한 줄이 이미지가 어디로 가는지 알려 준다(*Images are sent to api.anthropic.com*, *Images stay on this machine (localhost:11434)*).

### 로컬 모델 (YOLO · YOLOE · SAM)

- **장치**: *Local inference*에서 Automatic(CUDA > MPS > CPU) 또는 직접 지정. 메모리에 둘 모델 수(1–4)도 여기서 정한다. 모델의 첫 사용은 로딩·커널 컴파일로 10–20초 걸릴 수 있고, 이후에는 메모리에 남아 빠르다.
- **가중치**: `src/model`(`--model-dir`)에 있는 파일을 계열(yolo / yoloe / sam / text)별로 인식한다. Settings → *Model weights*에서 정해진 목록(yolo26n–x, yoloe-26s/m/l-seg, mobile_sam, sam2.1_t/s/b, mobileclip2_b.ts)만 **Download** 버튼으로 받는다. 자동 다운로드는 하지 않는다(`YOLO_OFFLINE`, `YOLO_AUTOINSTALL=false`). Ultralytics 가중치는 AGPL-3.0, SAM 계열은 Apache-2.0이다.
- **YOLO 클래스 매핑**: 모델 클래스는 명시한 매핑 → 이름(대소문자·복수형·동의어) 일치 순서로 데이터셋 클래스에 연결되고, **연결되지 않은 클래스는 버린다**(COCO ID를 그대로 쓰지 않음). `class_0…`처럼 이름 없는 데이터셋에서는 패널의 클래스 맵이나 클래스 카드를 먼저 채워야 한다. 한 검출 클래스가 여러 자세 클래스를 대신해야 하면(예: person → 서 있음/넘어짐) 2단계를 쓴다.
- **YOLOE**: *Text*는 클래스 카드의 프롬프트(없으면 이름)로 찾는다. 텍스트 인코더는 CPU에서 돌리고 임베딩을 캐시한다(MPS에서는 오히려 느렸다). *Your labels*는 클래스마다 라벨된 이미지 최대 8장의 시각 프롬프트 임베딩을 평균한 프로토타입으로 찾으며, 라벨 버전이 바뀔 때까지 캐시한다. 클래스를 따로 고르지 않았다면 라벨된 예시가 없는 클래스는 경고와 함께 빠진다.
- **SAM**: SAM2.1-t가 기본, 없으면 MobileSAM. 최근 3장의 이미지 특징을 캐시하므로 같은 이미지에서 반복 클릭이 빠르다.
- **배치 추론**: 일괄 작업과 감사는 로컬 모델에 8장씩(`options.infer_batch`, 1–32) 한 번에 넣고 다음 묶음을 미리 디코딩한다. MPS에서 yolo12m이 이미지당 호출 대비 약 2배(5.8–8.5 → 12.6–20.8 img/s), YOLOE-26s 예시 모드도 약 2배였고 결과 박스는 같았다.

### 클라우드 VLM (Claude · Gemini · OpenAI · OpenAI 호환)

모든 클라우드 공급자는 같은 파이프라인(`fovea/ai/vlm.py`)을 거친다.

- **모델마다 고유 좌표계로 요청한다**: Claude는 보낸 이미지의 픽셀 좌표, Gemini와 OpenAI는 0–1000 `[ymin, xmin, ymax, xmax]`, Qwen3 계열은 0–1000 `[x1, y1, x2, y2]`, Qwen2.5-VL은 28px 격자로 맞춘 이미지의 픽셀. 서버에서 공급자의 최대 크기로 미리 줄여 보내므로 좌표가 1:1로 맞는다. 0–1000과 픽셀을 혼동한 답은 감지해 고치고 경고를 남긴다.
- **구조화 출력**: JSON 스키마를 강제하고 데이터셋 클래스 이름을 닫힌 `enum`으로 넣는다. 클래스 카드의 설명·포함/제외 기준이 시스템 프롬프트에 들어간다(AI 탭 → Class cards의 *What a model is told*에서 확인).
- **노력 수준**: Fast / Balanced / Accurate(공급자에 low / medium / high로 보냄) 세 단계다. AI 패널, 배치 대화상자, Two-stage의 분류기, Benchmark의 각 설정에서 클라우드 모델을 고르면 나타나고, 따로 고르지 않으면 그 공급자의 Settings 기본값(*Effort*)을 쓴다. 로컬 모델에는 없다. 높은 노력은 mAP@50을 모델별로 +0.4~+13.6 올렸다(Gemini 3.6 Flash +13.6, Gemini 3.8 Flash +6.7, GPT-6.1 Sol +0.8, Qwen3.8 27B +0.4 — 두 점수가 모두 있는 `prices.json` 모델 기준) [Roboflow].
- **재시도**: `retry-after`를 따르고 408/409/429/5xx/529와 연결 오류는 지수 백오프로 재시도한다. 잡 상태 줄에 *Anthropic Claude rate limited — retrying in 7 s*처럼 표시되고, 취소하면 기다림도 즉시 끝난다. 429가 나면 공급자별 동시 요청 수를 절반으로 줄인다. 인증·할당량·잘못된 요청은 재시도하지 않고 잡을 멈춘다. 10장 연속 실패해도 멈춘다.
- **OpenAI 호환 서버**: Settings에서 *Server URL*과 *Box coordinates*(auto / qwen3 / qwen25 / gemini / norm)를 정한다. 서버 도달 여부를 백그라운드에서 확인해(`/v1/models`, 이어서 Ollama `/api/tags`) 60초 동안 기억하고, 꺼져 있으면 *Not reachable at <host>*로, 받아 두지 않은 모델은 *Not pulled on <host> — run: ollama pull …*으로 비활성화한다.

**기본·추천 모델** (가격은 1M 토큰당 입력/출력 USD, 2026-10-13 기준 `fovea/ai/prices.json`; mAP@50은 Roboflow Vision Evals 낮은/높은 노력, 2026-10-08)

| 모델 | mAP@50 | 가격 | 메모 |
| --- | --- | --- | --- |
| GPT-6.1 Sol (OpenAI 기본) | 80.8 / 81.6 | 2 / 10 | 이미지당 약 $0.01–0.02 |
| GPT-6 Astra | 82.1 / 83.6 | 10 / 50 | 가장 정확, 이미지당 $0.05–0.10 |
| Claude Sonnet 5.5 | 74.3 / 76.8 | 2 / 10 | Claude 중 박스가 가장 정확, 이미지당 약 $0.01 |
| Claude Opus 5.5 (Anthropic 기본) | 74.5 / 76.0 | 4 / 20 | 이미지당 약 $0.02–0.03 |
| Gemini 3.8 Flash (Gemini 기본) | 68.1 / 74.8 | 0.75 / 3.75 | 이 표에서 노력을 올릴 때 가장 많이 좋아짐(+6.7) |
| Claude Haiku 5.5 · GPT-6 Luna | 65.8 / 68.2 · 65.5 / 68.0 | 0.1 / 0.5 | 이미지당 $0.001 이하 — 2단계 분류·검증용 |
| Gemini 3.5 Flash-Lite | 57.5 | 0.3 / 2.5 | 약 4초, 이미지당 약 $0.002 |
| Qwen3.8 27B (Ollama, 호환 기본) | 65.7 / 66.1 | 0 | 로컬 최고, 메모리 약 20 GB |

박스에 쓰지 말 것: GPT-5.5(43.6), GPT-5.4 mini(15.8), Claude Opus 4.8 이하·Sonnet 5(36–39), Gemini 2.5 Pro(33.7). Claude Fable 5.1은 Sonnet 5.5보다 낮은 점수에 5배 가격이다. 모델 선택기에는 이 메모와 가격이 함께 표시된다.

## 2. 클래스 카드 — 모델과 사람이 공유하는 정의

Overview → Classes의 **Card** 또는 AI 탭 → **Class cards**에서 클래스마다 *prompt*(무엇을 찾을지), *definition*, *Box it when…*, *Don't box…*, 예시 박스(± 극성, 최대 12개)를 적는다. YOLOE 텍스트 모드와 모든 VLM 요청이 이 카드를 읽는다. 이름 없는 데이터셋(예: `class_0`–`class_11`)에서는 먼저 **Suggest names with AI**를 쓴다(6절).

## 3. 제안 검토 (Annotate · Explore · Objects)

**Annotate** (단축키 전체는 [keyboard.md](keyboard.md))

| 키 | 동작 |
| --- | --- |
| `I` / `⇧I` | 현재 모델로 이 이미지에 제안 요청 / 모델 선택 |
| `Enter` / `⇧Enter` | 선택한 제안 수락 / 보이는 제안 모두 수락 |
| `Del` / `⇧Del` | 제안 거절 / 보이는 제안 모두 거절 |
| `{` / `}` | 표시 임계값 −/+ 0.05 (기본 0.25, Explore와 공유). 패널 *Proposals* 칸의 `{` `}` 버튼도 같다 |
| `Tab` / `⇧Tab` | 보이는 제안 먼저, 그다음 박스 순회(포인터가 그림 밖이면 무언가 선택되어 있을 때만) |
| `⇧N` | 제안이 있는 다음 이미지 |
| `R` | 선택한 박스를 SAM으로 다듬어 제안으로 |
| `W` | 클릭으로 박스(SAM, `⇧`클릭은 제외 점) |
| `C` / `⇧C` | 이전 프레임 박스를 제안으로(IoU ≥ 0.7 중복 제외) / 바로 복사 |
| `K` | 키프레임 표시 → 두 키프레임 사이 프레임에 보간 제안(**Interpolate**). 제자리에서 클래스가 바뀐 박스(넘어짐)도 짝지음 |

- **`I`는 이미지마다 따로 돈다.** 클라우드 모델이 답하는 동안(5–30초) 다음 이미지로 넘어가도 요청은 계속되고, 결과는 요청한 그 이미지에 저장되어 돌아가면 보인다(*N proposals saved for <파일> — they show when you go back to it*). 그동안 패널 상태 줄에 *Proposing for <파일>*과 **Stop**이 남고, 지금 이미지에서 `I`를 누르면 이 이미지의 요청이 따로 시작된다. 동시에 3장까지 보내고, 넷째는 *Already proposing for 3 images*로 기다리게 한다. **Stop**은 기다림만 끝낸다: 모델이 그래도 답하면 제안은 그 이미지에 저장된다.
- 제안은 클래스 색 점선과 신뢰도 칩으로 그려진다. 제안을 끌거나 크기를 바꾸거나 숫자 키로 클래스를 바꾸면 **그 자체가 수락**이다.
- 같은 클래스의 기존 박스와 IoU ≥ 0.7인 제안은 *already labeled*로 숨는다. 패널 *Proposals* 칸의 한 줄이 *7 shown · 3 below 0.25 · 2 already labeled*처럼 숨은 수를 보여 주고, 숨은 제안(임계값 아래와 *already labeled*, 숨긴 클래스는 제외)은 하나씩 고를 수 없으므로 옆의 **Reject hidden**으로 한 번에 거절한다(`⌘Z`로 되돌림). 그 이미지에서 작업하고 떠나면 *already labeled* 제안은 자동으로 거절된다 — 수락하면 같은 물체에 박스가 하나 더 생길 뿐이라서.
- 다른 클래스의 박스와 IoU ≥ 0.7로 겹친 제안에는 ⇄가 붙는다(Annotate 목록의 *⇄ conflicts #n*). `⇧Enter`는 이런 제안을 수락하지 않고 남겨 두므로 `Enter`·`Del`로 하나씩 정한다(Annotate·Inspect 모두). Annotate의 `⇧Enter`는 신뢰도 높은 것부터 수락하면서, 방금 수락한 박스와 같은 클래스로 겹치는 제안은 중복으로 거절한다. 그래서 `⇧Enter`가 한 물체에 박스 두 개를 만들지 않는다.
- `⇧N`은 대기 중인 제안이 있는 다음 이미지로 가되, 그 제안이 모두 여기서 숨는 이미지(임계값 아래, 숨긴 클래스, *already labeled*)는 건너뛰고 *Skipped N images whose proposals are all hidden here*로 알린다. 한 번에 25장까지 확인하고, 그 안에 결정할 제안이 없으면 임계값을 낮추는 `{`나 **Reject hidden**을 안내한다.
- **`K` 보간**: 한 프레임에서 `K`로 키프레임을 표시하고 같은 시퀀스의 다른 프레임에서 `K`를 누르면 그 사이 프레임마다 보간 제안(≈)이 생긴다(한 번에 최대 2,000프레임, 같은 프레임에서 다시 `K`는 표시 해제). 두 키프레임의 박스는 1:1로 짝짓는다. 같은 클래스끼리는 어디로 움직였든 짝이 되고, 다른 클래스끼리는 **제자리에서만**(겹치거나 중심이 큰 박스 대각선의 0.75배 이내, 면적 차이 4배 이내) 짝이 된다 — 넘어짐 데이터셋에서는 클래스가 곧 자세라서, 사이에 넘어진 사람은 앞 키프레임에서 *서 있음*, 뒤 키프레임에서 *넘어짐*이기 때문이다. 비슷하게 가까운 같은 클래스 박스가 있으면 그쪽이 우선한다. 클래스가 바뀌는 박스의 보간 제안은 **가까운 키프레임의 클래스**를 받는다(정확히 가운데면 뒤 키프레임). 바뀐 순간은 알 수 없으므로 프레임마다 확인하라는 뜻으로 *reason*에 *class changes A → B: this frame takes the nearer keyframe's*가 남는다. 사이 프레임에 이미 그 물체의 박스(두 클래스 중 어느 쪽이든 IoU ≥ 0.5)가 있으면 건너뛴다. 이 비교도 1:1이라 기존 박스 하나는 보간 박스 하나만 대신한다 — 바짝 붙은 두 사람 중 한 명만 라벨되어 있어도 다른 한 명은 제안된다. 짝이 없는 박스는 보간하지 않는다. 결과 줄은 *1 box without a partner · 1 already labeled · 1 class change*처럼 빠진 것을 세고, Sequence 패널에 문장으로 남는다. 다음 `K`는 방금 누른 프레임을 키프레임으로 삼아 이어 간다. 한 프레임만 빠진 박스는 데이터 품질 칩 옆 **Fill from neighbours**가 더 빠르다([data-quality.md](data-quality.md#결과-쓰기)).
- 결정은 이미지를 떠날 때 서버에 저장되고, `⌘Z`는 제안 상태까지 되돌린다. 저장이 실패하면(네트워크, 서버 오류) 결정은 사라지지 않는다: 저장 상태가 *Decisions not saved*로 바뀌고 오류 토스트(8초에 한 번까지)가 뜨며, 4초·8초 뒤, 이후 16초마다 다시 보낸다(`⌘S`는 바로). 그동안 다른 이미지로 넘어갈 수 있고, 돌아오면 결정한 대로 보인다. 서버가 거부한 결정(408·429를 뺀 4xx)은 다시 보내지 않고 알린다. Annotate를 떠날 때 남은 결정은 한 번 더 보내 보고, 그래도 안 되면 *N proposal decisions were not saved — they show as pending again*으로 알린다.
- **자동화 편향 대책**: 미리 수락된 제안은 없다. `⇧Enter`는 지금 보이는 것만 수락하고, 수락한 것 중 신뢰도가 낮은 3분의 1에서 한 박스를 골라 *spot-check*로 선택해 둔다. 200개 넘게 아무 수정 없이 98% 이상을 수락하면 임계값을 낮춰 표본 검사하라는 정적인 안내가 뜬다. 근거: 제안 확인은 박스를 그리는 것보다 15–25배 빠르고(1.6초 대 26–42초) [Papadopoulos 2016], 재현율 높은 AI와 한 키 거절 조합이 가장 빠르고 정확했으며 [Xu 2023], 수정이 번거로울수록 사람은 틀린 제안을 더 받아들였다 [Beck 2025].

**Explore / Inspect**: 타일 배지 ◌ n(대기 중인 제안), 필터 *Has AI proposals*, 정렬 *Lowest AI confidence*. Inspect에서 `⇧Enter`로 보이는 제안 수락, `Delete`로 거절, `I`로 **Ask AI to check the labels**(이 이미지 하나를 감사). 선택 일괄 메뉴의 *AI proposals on the selection*에서 *Accept proposals ≥ 0.25*(현재 임계값), *Reject pending proposals*, *Reject proposals below 0.25*. 거절한 제안은 지우지 않고 남겨 같은 제안이 다시 뜨지 않게 한다.

**일괄 수락의 규칙**(Explore의 *Accept proposals*, Inspect `⇧Enter`, AI 탭 → Label images의 대기 제안 수락)은 Annotate `⇧Enter`와 같다. 신뢰도 높은 것부터 보면서, 같은 클래스 박스와 IoU ≥ 0.7로 겹치는 제안은 *already labeled*로 아무것도 쓰지 않고, 다른 클래스 박스(파일에 있던 것이나 방금 수락한 것)에만 겹치는 제안(⇄)은 쓰지 않고 대기로 남겨 Annotate에서 정하게 한다. 한 물체에 클래스가 둘 붙지 않게 하기 위해서다. 결과에 *N already labeled*, *N ⇄ left to decide (over another class)*로 나온다. 일괄 수락은 쓰기 전에 스냅샷을 남기므로 Overview → History의 **Restore**로 되돌리면 라벨 파일과 함께 그 작업이 수락한 제안도 다시 대기 상태가 된다(*N AI proposals pending again*). 라벨 파일을 되돌리지 않고 남긴 이미지(되돌리는 사이 다른 곳에서 저장된 파일, 실패한 파일)의 제안은 결정된 그대로 둔다. Inspect에서는 `⌘Z`가 그 이미지 하나의 수락을 같은 방식으로 되돌린다.

**Objects**: AI 감사 의심 박스에 ⇄ ⤢ ✕ 표시, `A`로 수락(클래스 바꿈·삭제), 느슨한 박스는 제안 박스를 점선으로 보여 주고 `Enter`로 Annotate에서 맞춘다. `⌘Z`는 파일과 제안 상태를 함께 되돌린다.

## 4. 일괄 라벨링 (Batch)

Annotate의 **Batch…** 또는 AI 탭 → **Label images**.

- **범위**: Annotate에서는 여기서부터 N장 / 현재 목록 / 이 이미지 / 붙여 넣은 ID, AI 탭에서는 라벨 파일 없는 이미지 / 박스 없는 이미지 / 전체 / 라벨된 이미지 / ID 목록 + 분할·무작위 표본. Explore에서 *Label the selection with AI…*로 넘어오면 선택(`?ids=` 또는 `?set=`)이 그대로 범위가 된다.
- **대상**: **Proposals**(기본, 라벨 파일을 건드리지 않음, 이미지의 이전 대기 제안은 교체) · **Fill empty**(라벨 파일이 없는 이미지에만 씀 — 빈 라벨 파일은 의도한 배경으로 보고 건너뜀) · **Append**(기존 박스 옆에 추가. 같은 클래스 박스와 IoU ≥ 0.7로 겹치는 박스는 중복으로 건너뛰고, 다른 클래스 박스에 겹치는 박스는 쓰지 않고 이 실행의 대기 제안(⇄)으로 남겨 Annotate에서 정하게 함 — 결과에 *N duplicates skipped*, *N over a box of another class left to decide in Annotate*) · **Replace**(기존 박스 삭제 후 씀).
- **쓰기 안전**: 쓰기 대상은 먼저 서버가 정확한 영향 범위(예: *9 existing boxes on 3 labeled images are deleted…*)를 계산해 보여 준다. 20장 넘는 Replace는 확인 문구를 직접 입력해야 한다. 실행 직전 스냅샷을 남기고, 끝나면 **Undo these changes**로 복원한다. Undo는 이 실행이 쓴 파일만, 그 파일이 실행이 남긴 내용 그대로일 때만 되돌린다: 그 뒤 Annotate 등에서 고친 파일은 그대로 두고 *N edited since — kept*로 알리며, *Fill empty*가 건너뛴 이미지에 나중에 직접 만든 라벨 파일도 지우지 않는다. 버전 전체를 되돌리려면 Overview → History의 **Restore**를 쓴다. 모델이 아무것도 못 찾았다고 라벨 파일을 새로 만들거나 비우지 않는다. confidence 열은 *keep confidence*를 켰을 때만 남긴다.
- **추정과 한도**: 시작 전 예상 비용(범위)·Economy 가격·토큰·이미지당 시간·가정이 보인다. 실행별 예산(budget)을 넘으면 깔끔하게 멈추고, 공급자의 남은 30일 지출 상한(진행 중인 다른 실행이 예약한 몫을 뺀 것, 9절)도 예산처럼 작동한다(이미 상한에 닿았으면 시작을 거부). 진행 중에는 비용·처리 수·박스·오류와 경과 시간, **Cancel**이 보인다.
- **상한 없는 실행은 한 번 더 묻는다**: 클라우드 실행에 예산도 공급자 상한도 없으면 끝날 때까지 아무것도 멈추지 않는다. 예상 비용의 높은 쪽이 $5 이상이면 대화상자에 *No spend cap: nothing stops this run before it ends…* 줄이 보이고, **Start**를 누르면 *Start a cloud run with no spend cap?*로 확인한다(**Set a cap**은 예산 칸으로 돌아감, Economy는 *Submit without a cap*). 쓰기 대상이면 같은 내용이 쓰기 확인 창 안에 들어간다. 가격 정보가 없는 모델은 20장 이상이면 *Start a run whose cost is unknown?*로 묻는다(9절).
- **미리보기**: *Preview on 3 images*로 내 라벨 위에 AI 박스를 점선으로 겹쳐 본다. 미리보기 자체는 따로 추정하지 않고 바로 보내며, 결과에 이미지별 실제 비용이 붙는다.

### Economy · −50% (공급자 배치 API)

배치 API가 있는 클라우드 공급자(Anthropic Message Batches, OpenAI `/v1/batches`, Gemini `batchGenerateContent`)를 고르면 *Speed*에 **Now / Economy · −50%**가 나타난다. Economy는 결과를 24시간 안에 받는 대신 약 절반 가격이다.

- 1,000장 단위로 제출하고(공급자 한도에 따라 더 나뉨), 60초마다(`FOVEA_AI_BATCH_POLL_S`) 상태를 확인한다. 기다리는 동안 AI 워커를 점유하지 않는다.
- 결과는 Now와 똑같이 제안 또는 쓰기 대상(스냅샷 후)으로 들어가고 배치 가격으로 기록된다.
- **서버를 재시작해도** 공급자 배치는 계속 돌고, 시작할 때 진행 중인 Economy 실행에 다시 붙는다. 제출 도중 서버가 멈추면 아직 보내지 않은 이미지는 다시 시작해도 보내지 않는다 — 공급자가 받았는지 알 수 없는 묶음을 다시 보내면 두 번 과금될 수 있어서다. 그 수는 잡 결과와 Runs & cost에 *N never sent*로 표시되니(API 필드는 `not_submitted`), 필요하면 그 이미지에 AI를 다시 돌린다.
- **취소**는 AI 탭 → Runs & cost의 *Provider batches in progress* 카드나 잡 상자에서 한다. 공급자 배치도 취소되며, 이미 끝난 결과는 저장된다(그 요청은 어차피 과금된다).
- 2단계 파이프라인은 Economy를 쓸 수 없고, `refine` 옵션은 건너뛴다. 디코딩 정보는 `<data-dir>/ai_batches`에 40일 보관된다(키·픽셀 없음).
- Settings → AI providers → OpenAI의 **Flex tier** 스위치(*Cheaper, slower*)는 Economy와 별개다. **OpenAI에만** 있고, 켜면 OpenAI 검출·분류 요청(한 장짜리 제안, Inspect의 라벨 검사, 라벨 감사, Now 일괄 실행, 2단계 분류 포함)을 Flex 등급으로 보낸다: 배치 가격이지만 답이 몇 분 걸릴 수 있고, Flex 등급이 없는 모델은 표준 가격으로 돈다(실제 응답의 등급대로 과금 기록). 켜 두면 추정의 Now 가격도 배치 가격으로 계산되고(가정 목록에 *the Flex tier* 표시) 예산 확인도 그 가격을 쓴다. 그래서 배치 대화상자의 Economy는 −50% 표시 없이 *the same price as the Flex tier*로 나온다 — 더 싸지지 않고 24시간까지 기다리기만 한다. Ask AI(자연어 검색)와 Suggest names는 항상 표준 등급이다. Anthropic·Gemini에는 이 스위치가 없고, *Batch API* 줄이 배치 대화상자의 Economy를 안내한다(OpenAI에도 같은 줄이 있다).

## 5. 2단계 검출 + 분류 (넘어짐 같은 자세 클래스)

VLM은 분류에 강하고 정밀한 박스에 약하며(최신 모델도 mAP@50:95는 55–62 수준) [Roboflow], 로컬 검출기는 그 반대다. 검출기가 제안하고 VLM이 검증하는 조합에서 GPT-4o 검수자는 사람과 약 85% 일치했다 [DART]. 그래서 넘어짐 데이터셋처럼 한 물체(사람)가 여러 상태 클래스로 나뉘면 AI 탭 → **Two-stage**를 쓴다.

1. **Detector**: 로컬 YOLO(예: yolo12m의 `person`)나 YOLOE 텍스트가 박스를 찾는다. 신뢰도는 검출기의 것을 쓴다.
2. **Classifier**: 각 박스를 25% 여백(15/50% 선택 가능)을 두고 잘라 박스를 빨간 테두리로 표시한 뒤(긴 변 336–448 px), 한 요청에 최대 20개씩 VLM에 보내 데이터셋 클래스 중 하나 또는 *none/uncertain*으로 답하게 한다. 이유와 분류 신뢰도는 제안의 *reason*에 남는다(예: *person → class_1*).
3. **Test on 4 labeled images**로 내 라벨과의 일치(agree / differ / not labeled / missed), 이미지당 비용·시간을 본 뒤 *Use in Annotate* 또는 *Run on images…*.

## 6. 클래스 이름 제안 · 자연어 검색

- **Suggest names with AI**(Overview → Classes, AI 탭 → Class cards): 클래스마다 전형적인 박스(크고 중앙에 있으며 서로 다른 시퀀스·이미지에서) 4/6/8개를 384 px 크롭에 빨간 테두리로 보내 *name*(snake_case 영어), *prompt*, *definition*, 신뢰도를 받는다. 크롭이 20장 이하면 한 요청, 많으면 클래스별 요청(최대 4개 병렬). **실행 전 정확한 예상 비용**을 보여 주고(아무것도 보내지 않는 dry run), 결과는 행마다 편집·적용하며 **Apply**는 클래스 목록과 카드의 prompt/definition을 함께 쓴다. 행마다 **Revert** 가능. 이름 중복·빈 이름은 적용을 막는다.
- **Ask AI**(Explore 검색창): 한국어·영어 질문을 Explore 필터(분할, 클래스, 이슈, 리뷰·AI 상태, 박스 수, 정렬)로 바꾼다. 질문과 클래스 이름·카드 프롬프트·분할·이슈 코드만 보내고 이미지는 보내지 않는다. 공급자의 가장 싼 모델(예: Claude Haiku 5.5, GPT-6 Luna)을 쓰며, 모르는 키·클래스·분할은 서버가 버리고 경고로 알려 준다. 같은 질문은 10분간 캐시되어 무료다.
- 둘 다 *generate*가 가능한 공급자(Claude, Gemini, OpenAI, OpenAI 호환 서버)가 필요하다. 로컬 검출기만 있으면 400 `unavailable`.

## 7. 라벨 감사 (Label audit)

AI 탭 → **Label audit**. 검출기를 낮은 신뢰도 하한(기본 0.05)으로 돌려 내 라벨과 비교하고, 라벨이 틀렸을 가능성이 높은 순으로 이미지를 정렬한다(ObjectLab 방식 [Cleanlab]).

| 종류 | 표시 | 조건 (기본 `hi` = 0.5) | 한 키 수정 |
| --- | --- | --- | --- |
| `ai_missing` 빠진 객체 | + | 신뢰도 ≥ hi인 검출을 같은 클래스 박스가 IoU 0.3 이상 덮지 않고, 어떤 박스와도 IoU < 0.5 | 박스 추가 |
| `ai_loose` 느슨한 박스 | ⤢ | 같은 클래스 검출과의 최고 IoU가 0.3–0.8, 그 검출 신뢰도 ≥ hi | 제안 박스로 맞춤 |
| `ai_swap` 클래스 뒤바뀜 | ⇄ | 다른 클래스 검출(≥ hi)과 IoU ≥ 0.5인데 같은 클래스와는 < 0.3 | 클래스 변경 |
| `ai_spurious` 근거 없음 | ✕ | 어떤 검출과도 IoU < 0.3 | 박스 삭제 |

- 이미지 점수는 소견 점수의 softmax 풀링(T = 0.1): 가장 나쁜 소견이 지배하지만 중간 소견 여럿도 더해진다.
- **모드**: *Geometry*는 클래스 묶음(예: COCO `person` → 내 클래스 [0, 1])을 하나로 보고 위치만 따진다 — 자세 클래스처럼 검출기가 구분 못 하는 경우. *Geometry + classes*(full)는 클래스까지 따진다(닫힌 어휘 모델은 클래스 매핑 필요). 검출기가 볼 수 없는 클래스는 판정하지 않는다.
- 보조 모델 최대 3개를 *committee*로 더할 수 있다. 라벨을 만든 모델과 다른 계열을 쓰면 확증 편향이 줄어든다.
- **검토**: Overview → AI label audit 카드, Explore `?ai=audit&sort=audit&order=desc`(가장 의심스러운 것부터 — Overview·AI 탭의 링크가 모두 이렇게 열고, 정렬 없이 `issue=ai_*`나 `ai=audit`로 걸러도 Explore·Annotate가 같은 순서로 연다), `J`/`⇧J`로 다음 라벨 문제 이미지(AI 소견, 라벨 파일 이슈, 시퀀스·모양 플래그), Annotate의 기호와 `Enter` 수정, Objects의 `A`. 소견은 원래 GT 박스 좌표(`ref_box`)를 함께 저장하므로 위쪽 박스를 지워 행 번호가 바뀌어도 맞는 박스를 가리킨다. 이후 라벨이 바뀐 소견은 적용 대신 거절된다.
- 감사 실행은 검출기의 원래 검출도 저장하므로 **Compare에서 감사 실행 자체를 평가**할 수 있다(기하 모드는 묶음 단위로 채점).

## 8. 벤치마크

AI 탭 → **Benchmark**. 2–4개 설정(A–D: 공급자·모델·노력)을 라벨된 이미지 무작위 표본에 돌려 정밀도·재현율·F1·AP50(신뢰도가 있을 때)·이미지당 시간·이미지당 비용·오류를 표로 비교하고, 클래스별 AP도 본다. IoU는 0.50 또는 0.75. **Use**로 이긴 설정을 기본으로 삼는다. 예측은 저장되어 Compare에서 자세히 볼 수 있다. 공개 순위가 도메인마다 크게 달라지므로(예: RF100-VL에서 VLM의 제로샷 mAP는 5–12) [RF100-VL] 실제 데이터에서 재는 것이 기본이다.

## 9. 비용

- **추정**: 공급자별 이미지 토큰 계산(해상도 등급), 프롬프트, 노력별 출력 범위, 타일·샘플 수를 곱해 `usd`(범위)와 `usd_batch`(Economy)를 낸다. 이 PC에서 잰 처리 시간이 있으면 그것을 쓴다. 2단계는 두 단계를 모두 더한다.
- **어디서 보이나**: 일괄 라벨링(Now·Economy), Two-stage 설정(100장당), 라벨 감사, 벤치마크(설정별), 클래스 이름 제안(dry run)은 시작 전에 추정을 보여 준다. 한 장짜리 호출 — Annotate `I`(`/api/ai/propose`), Inspect의 **Ask AI to check the labels**(`/api/ai/review`), Explore **Ask AI**(`/api/ai/nl-filter`), *Preview on 3 images*·*Test on 4 labeled images*(`/api/ai/preview`) — 는 추정 없이 바로 보낸다. 모델 선택기의 1M 토큰당 가격이 기준이고, 실제 비용은 결과 옆과 Runs & cost에 남는다.
- **한도**: 실행별 예산(`budget_usd`)은 일괄 라벨링에만 있다. 공급자별 30일 롤링 지출 상한(Settings → *Spend cap*)에 닿으면 새 실행과 한 장짜리 호출을 모두 거부하고, 일괄 라벨링과 라벨 감사는 실행 도중에도 상한에서 멈춘다.
- **예약**: 아직 과금되지 않았지만 이미 약속된 돈도 상한에서 뺀다. 공급자에서 기다리는 Economy 실행은 보낸 이미지 × 배치 가격의 높은 추정(이미 과금된 몫 제외)을, 진행 중인 Now 실행은 남은 이미지의 몫을 예약한다. 그래서 동시에 시작한 두 실행이 같은 남은 상한을 각자 다 쓰지 않는다. Settings의 사용 줄에 *spent $0.23 in the last 30 days · reserved by runs still going $0.92*처럼 보이고, 예약으로 상한이 차면 상태가 *Spend cap held by runs still going*이 되며 새 실행은 *(… is held by runs still going)*과 함께 거부된다.
- **상한 없는 실행의 확인**: 예산도 30일 상한도 없는 클라우드 일괄 라벨링과 라벨 감사(감사에는 실행별 예산이 없음)는 예상 비용의 높은 쪽이 $5 이상이면 시작 전에 *Start a cloud run / audit with no spend cap?*로 한 번 더 묻는다. 한 번의 클릭이 수백 달러를 쓰지 않게 하려는 것이다. 공급자에 상한을 걸어 두면 묻지 않는다.
- **가격 정보가 없는 모델**: `prices.json`에 없는 클라우드 모델(Settings의 **Refresh list**로 새로 생긴 모델)은 비용을 셀 수 없어 $0으로 기록되고 추정에 *cost unknown*으로 나온다. 그래서 실행별 예산이나 그 공급자의 30일 상한이 걸려 있으면 그 모델의 호출은 한 장짜리까지 모두 거부되고(*No price on file for …*), 상한이 없으면 20장 이상의 일괄 실행·감사에서 *whose cost is unknown?*으로 묻는다.
- **기록**: AI 탭 → **Runs & cost**에 실행마다 토큰·비용·상태가 남고, 30일 일별 차트와 공급자·모델별 합계가 보인다. 돈이 든 실행을 지워도 사용 기록은 남는다. 클래스 이름 제안과 Ask AI 검색도 기록된다.
- **실패해도 과금된 답**: 모델이 거절한 답(*declined to process this image*), JSON으로 읽을 수 없는 답, 출력 한도에서 잘린 답은 공급자가 토큰을 받아 간다. 그래서 그 비용도 실행의 비용·사용 합계·실행별 예산·30일 지출 상한에 그대로 들어가고, 처리한 이미지 수에는 넣지 않는다. 한 장짜리 호출(제안, 라벨 검사, 미리보기, Ask AI)이 이렇게 실패해도 그 비용은 Runs & cost에 남는다(제안·라벨 검사는 *error* 상태의 실행으로). 타일·2단계 분류·위원회처럼 한 이미지에 여러 번 호출하다 중간에 실패하면 앞서 성공한 호출의 비용도 함께 기록된다. 잘린 답은 먼저 더 큰 출력 한도로 한 번 다시 묻고(두 번 모두 과금), 그래도 잘리면 완성된 박스만 쓴다(하나도 없으면 오류).
- 가격은 `fovea/ai/prices.json`(기준일 2026-10-13, 출처: [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) · [Gemini](https://ai.google.dev/gemini-api/docs/pricing) · [OpenAI](https://developers.openai.com/api/docs/pricing))에 있다. 공급자 모델 목록을 새로 받아도 가격은 이 표에서만 온다.

## 10. 키 · 보안 · 개인정보

- **키 저장**: 환경 변수(`ANTHROPIC_API_KEY`, `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `OPENAI_COMPAT_API_KEY`)가 우선이고, Settings에 입력한 키는 `<data-dir>/secrets.json`(0600, 원자적 쓰기)에 저장된다. 환경 변수 키는 Settings에서 읽기 전용으로 보인다.
- **노출 없음**: API는 키의 있음 여부·출처·끝 4자리만 돌려준다. 오류 메시지·로그·잡 결과·실행 파라미터에서 키 모양 문자열(`sk-ant-…`, `sk-…`, `AIza…`, Bearer 토큰, 저장된 키 값)을 지운다. 키는 URL에 넣지 않는다(Gemini도 헤더 사용).
- **게이트웨이 주소의 자격 증명**: `FOVEA_<PROVIDER>_BASE_URL`(예: `FOVEA_OPENAI_COMPAT_BASE_URL=https://user:password@gateway.example/v1`)에 `user:password@`를 넣으면 요청에 basic auth로 붙는다. 이 주소는 요청을 만들 때만 쓰고, 공급자 상태·위치 줄·모델 목록·오류 메시지·로그·저장된 배치 정보에는 `host:port`만 나온다. Settings의 *Server URL*에는 자격 증명이 든 주소를 넣을 수 없다(*Put credentials in the key field, not in the URL*) — 키 칸을 쓴다.
- **원격 차단**: 다른 PC에서 온 요청은 키를 바꾸거나 클라우드 모델을 호출할 수 없다(`FOVEA_ALLOW_REMOTE_AI=1`로 해제). 저장된 키를 보내는 OpenAI 호환 서버도 같다 — 키가 붙은 주소는 유료 게이트웨이일 수 있어서다(키 없는 로컬 서버와 YOLO·YOLOE·SAM은 그대로). 브라우저가 다른 PC인지는 서버가 판단해(`GET /api/meta`의 `request_is_local`, 실패하면 주소창의 호스트 이름으로 대신) 화면에 반영한다: AI 탭 머리에 *Cloud AI off from this browser* 칩과 안내, Annotate 패널·모델 선택기의 막힌 항목은 *Cloud AI is off for browsers on other machines*(키를 보내는 호환 서버는 *Uses the server’s saved API key — off for browsers on other machines*)로 비활성, Explore Ask AI는 그 공급자를 쓰지 않음, Settings → AI providers는 *You are using Fovea from another machine* 안내와 함께 읽기 전용. "이 PC"는 접속한 주소(루프백 또는 `FOVEA_LOCAL_NETS`)로 판단하므로 Docker를 다른 PC에 열면(`FOVEA_LOCAL_NETS`를 지우면) Docker를 실행한 PC의 브라우저도 다른 PC로 취급된다. 그때 클라우드 AI를 쓰려면 `src/.env`에 `FOVEA_ALLOW_REMOTE_AI=1`을 넣고 `docker compose up -d`로 컨테이너를 다시 만든다(README의 *Docker*).
- **무엇이 전송되나**: 검출·2단계·감사·벤치마크는 이미지(또는 박스 크롭)와 클래스 이름·카드 내용을 공급자에게 보낸다. 클래스 이름 제안은 박스 크롭을, Ask AI는 질문 텍스트와 클래스 정보만 보낸다. OpenAI 요청은 `store: false`로 보낸다. 로컬 모델과 루프백 주소의 OpenAI 호환 서버는 이미지를 이 PC 밖으로 보내지 않는다.

## 11. API 전용 옵션

UI에 아직 컨트롤이 없지만 `POST /api/ai/propose|preview|batch|estimate`의 `options`로 쓸 수 있다.

| 옵션 | 공급자 | 설명 |
| --- | --- | --- |
| `tiles: true \| "auto" \| n \| [cols, rows]` | 클라우드·OpenAI 호환 | SAHI식 타일 + 전체 이미지 한 번, NMS로 병합. 작은 물체가 많은 큰 이미지용. `n` = n×n 격자(1–4), `[cols, rows]` = 각 1–4, `"auto"` = 이미지가 모델 입력보다 훨씬 클 때만 격자 |
| `tiles: true \| "auto" \| n` | YOLO · YOLOE | 같은 방식이지만 `n`은 타일 한 변의 픽셀 크기(64–4096). `true`/`"auto"` = 모델 입력 크기(`imgsz`) 타일 |
| `samples: 1–5` | 클라우드 | 같은 요청을 N번 보내 IoU로 맞춰 신뢰도 = 표 수 / N. VLM은 보정된 신뢰도를 주지 않으므로 Compare의 AP에 필요 |
| `refine: true \| "auto" \| "all" \| {max_boxes, iterations}` | 클라우드 | 작은 박스(면적 2% 미만)나 샘플끼리 어긋난 박스를 2.5배 문맥 크롭으로 다시 묻는다. IoU ≥ 0.25, 면적 비 0.5–2, 중심 이동 ≤ 0.35일 때만 채택 [LFPR], 최대 2회 [C2F-Space]. 비용 상한이 올라간다 |
| `ruler: true` (실험) | Gemini · OpenAI · 0–1000 좌표의 호환 모델 | 위·왼쪽 여백에 눈금자를 그려 보내고 디코딩 때 여백을 뺀다. SOTIF 연구에서 Gemini 3 mAP50 0.525 → 0.695 [SOTIF]. Claude 같은 픽셀 좌표 모델에서는 경고와 함께 건너뜀 |
| `infer_batch: 1–32` | YOLO · YOLOE | 일괄 작업에서 한 번에 추론할 이미지 수(기본 8) |

## 12. 문제 해결

| 증상 | 원인과 해결 |
| --- | --- |
| 클라우드 모델이 회색 *Add key in Settings…* | 키가 없음. Settings → AI providers에 키 저장 후 **Test** |
| *Cloud AI providers and API keys can only be used from the computer Fovea runs on* (403), *Cloud AI off from this browser*, *Cloud AI is off for browsers on other machines* | 서버가 다른 PC로 보는 브라우저. 서버 PC에서 `localhost` 주소로 쓰거나, 서버를 `FOVEA_ALLOW_REMOTE_AI=1`(또는 `--allow-remote-ai`)로 시작. **Docker를 다른 PC에 열었다면**(`FOVEA_LOCAL_NETS`를 지움) 서버 PC의 브라우저도 다른 PC로 보이므로 서버 PC에서 써도 같은 403이 난다: `src/.env`에 `FOVEA_ALLOW_REMOTE_AI=1`을 넣고 `docker compose up -d` |
| YOLO가 아무 박스도 안 만듦 / 422 | 모델 클래스가 데이터셋 클래스에 매핑되지 않음. 패널의 클래스 맵, 클래스 카드, 또는 2단계 사용 |
| *Not reachable at localhost:11434* | Ollama 등 서버가 꺼져 있음. 켜면 60초 안에 자동 갱신. *Not pulled …* 는 안내된 `ollama pull` 실행 |
| YOLOE 텍스트 모드 오류 | `mobileclip2_b.ts`가 없음 → Settings → Model weights에서 Download, 또는 *Your labels* 모드 사용 |
| 첫 제안이 10–20초 | 모델 로딩과 커널 컴파일. 이후는 빠르다. *Local inference*의 장치 확인(CPU면 수 배 느림) |
| *rate limited — retrying in N s* | 공급자 속도 제한. 기다리면 이어서 진행. 자주 나면 *Parallel requests*를 낮춤 |
| 잡이 *quota*/*auth*로 멈춤 | 지출 한도·잔액 또는 잘못된 키. 공급자 콘솔 확인 후 키 교체 |
| Economy가 안 보임 | 배치 API 없는 공급자(로컬·호환 서버)이거나 2단계 파이프라인 |
| Economy 실행이 오래 *waiting* | 공급자가 처리 중(최대 24시간). Runs & cost에서 상태·다음 확인 시각 확인, 필요하면 Cancel |
| 거의 모두 수락했다는 안내가 뜸 | 자동화 편향 경고. 임계값을 낮춰 표본을 직접 확인 |
| Economy 실행에 *N never sent* | 제출 도중 서버가 멈춤. 이중 과금을 피하려고 다시 보내지 않으므로 그 이미지에 AI를 다시 돌림 |
| **Undo these changes** 후 *N edited since — kept* | 실행 뒤 손으로 고친 파일은 Undo가 덮어쓰지 않음. 그것까지 되돌리려면 Overview → History의 **Restore** |
| 일괄 수락·Append 뒤에도 ⇄ 제안이 남음(*N over a box of another class left to decide in Annotate*) | 다른 클래스 박스와 겹친 제안은 한 물체에 클래스가 둘 붙지 않도록 쓰지 않고 대기로 남긴다. Annotate에서 하나씩 `Enter`·`Del`로 정함 |
| *Spend cap held by runs still going* / 새 실행이 *… is held by runs still going*으로 거부됨 | 진행 중인 실행(기다리는 Economy 배치 포함)이 예약한 몫으로 상한이 참. 그 실행이 끝나 실제 비용이 정해지면 풀림. 급하면 그 실행을 Cancel하거나 상한을 올림 |
| Annotate 저장 상태가 *Decisions not saved* | 수락·거절 결정이 서버에 닿지 않음(네트워크·서버). 결정은 남아 있고 백그라운드에서 다시 보냄. `⌘S`로 바로 재시도 |
| Start를 누르면 *Start a cloud run with no spend cap?* | 예산도 30일 상한도 없는데 예상 비용의 높은 쪽이 $5 이상. 대화상자의 예산 칸이나 Settings의 *Spend cap*을 채우면 묻지 않음 |
| `⇧N`이 ◌ 표시가 있는 이미지를 건너뜀(*Skipped N images…*, *No proposals to decide in the next N images…*) | 그 이미지들의 제안이 모두 임계값 아래이거나 이미 라벨된 박스와 겹침. `{`로 임계값을 낮추거나 **Reject hidden** |

## 근거

- [Roboflow] Roboflow Vision Evals, Object Detection (2026-10-08, 같은 표본·고정 클래스 목록·3회 평균): https://playground.roboflow.com/evals/object-detection
- [Papadopoulos 2016] Papadopoulos et al. *We Don't Need No Bounding-Boxes.* CVPR 2016. https://arxiv.org/abs/1602.08405
- [Xu 2023] Xu, Lien, Höllerer. *Comparing Zealous and Restrained AI Recommendations in a Real-World Human-AI Collaboration Task.* CHI 2023. https://arxiv.org/abs/2410.11860
- [Beck 2025] Beck, Eckman, Kern, Kreuter. *Bias in the Loop: How Humans Evaluate AI-Generated Suggestions.* arXiv:2509.08514; HDSR 2026. https://arxiv.org/abs/2509.08514
- [Cleanlab] ObjectLab: automated QA for object detection datasets (2023). https://cleanlab.ai/blog/object-detection/ · arXiv:2309.00832
- [RF100-VL] https://arxiv.org/abs/2505.20612
- [SOTIF] LVLM detection under long-tail and degradations. https://arxiv.org/abs/2601.22830
- [LFPR] https://arxiv.org/abs/2608.19553 · [C2F-Space] https://arxiv.org/abs/2511.15333
- [DART] *DART: An Automated End-to-End Object Detection Pipeline with Data Diversification, Open-Vocabulary Bounding Box Annotation, Pseudo-Label Review, and Model Training* (2024, n = 50 이미지). https://arxiv.org/abs/2407.09174
- Anthropic 좌표 가이드(픽셀 좌표 권장): https://platform.claude.com/docs/en/build-with-claude/vision-coordinates · Gemini 공간 이해 쿡북: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb · Qwen3-VL 2D grounding: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb
- YOLOE: https://docs.ultralytics.com/models/yoloe/ · SAM 2: https://docs.ultralytics.com/models/sam-2/ · SAHI: https://arxiv.org/abs/2202.06934
