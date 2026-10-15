# YOLO가 아닌 어노테이션 형식

데이터셋을 추가할 때 라벨이 YOLO `.txt`가 아니면(VisDrone의 픽셀 CSV, KITTI, DOTA, Pascal VOC `.xml`, COCO·LabelMe `.json`, CVAT `.xml`, 또는 목록에 없는 형식) Fovea는 그 형식을 알아내고, 실제 파일로 검증한 뒤, 원본은 그대로 두고 YOLO 라벨로 변환해 인덱싱한다. 형식이 가진 박스별 속성(occlusion, truncation …)과 객체가 아닌 영역(VisDrone의 ignored regions, KITTI의 DontCare, COCO의 crowd)도 함께 가져와 Annotate·Explore·Objects·AI에서 쓴다. 구현: `src/fovea/core/annformats.py`(형식 카드·판별·검증·변환), `src/fovea/core/annindex.py`(인덱스·속성·영역), `src/fovea/ai/formats.py`(AI·웹 조회).

## 형식 카드

형식마다 **형식 카드**가 있다: 파일 종류, 한 줄(또는 요소)을 박스로 읽는 **읽기 규칙**(구분자, 좌표 열, 박스 표현 `ltwh`·`xyxy`·`cxcywh`·`poly`, 픽셀/정규화 단위, 클래스 열과 값→클래스 매핑), 공식 클래스 이름과 정의, 속성(열과 값의 의미), 객체가 아닌 영역, 공식 링크·인용·라이선스. 카드는 데이터셋 레이아웃(`layout.formats`)과 변환 폴더의 `.fovea-format.json`에 저장된다.

## 판별 순서

1. **YOLO인지 먼저 본다**: 라벨 후보 폴더의 `.txt` 표본 줄이 YOLO 행(공백 구분 숫자, 정수 클래스, 0~1 좌표)이면 그대로 읽는다. 아니면 라벨 폴더로 쓰지 않는다 — AI 폴더 찾기가 다른 형식의 폴더를 라벨이라고 답해도 마찬가지다.
2. **내장 목록**(AI·네트워크 없음): VisDrone-DET, KITTI, DOTA, Pascal VOC, LabelMe, COCO, CVAT for images. 파일 생김새로 판별한다(예: VisDrone은 쉼표로 구분된 정수 8칸, 6번째 칸이 0~11).
3. **같은 데이터셋의 다른 분할이 이미 쓰는 카드**.
4. **AI**(Settings → AI → *Read annotation formats*, 기본값 *Built-in formats, then AI with the official documentation*). 한 줄에 한 객체인 `.txt` 형식만 AI가 읽고, `.xml`·`.json`은 내장 목록만 쓴다.
   1. 가장 싼 모델이 폴더 이름(고른 폴더 이름이 `data`·`raw`처럼 일반적이면 상위 폴더 이름까지, 예: `CARPK_devkit/data`)과 표본 파일 3개(각 10줄, 이미지 크기 포함)를 보고 **공개된 데이터셋인지** 판단한다.
   2. 공개 데이터셋이면 웹 검색이 되는 공급자(Claude·Gemini·OpenAI)의 기본 모델이 공식 문서(사이트, GitHub 툴킷, 논문)를 찾아 그 정의대로 읽기 규칙, 클래스 이름·정의, 영역, 속성 의미를 쓰고 근거 URL을 남긴다. Hugging Face 데이터셋 카드도 찾아 참고한다.
   3. 공개 데이터셋이 아니거나 웹 조회가 안 되면 기본 모델이 표본만으로 규칙을 쓴다.
   4. 규칙이 검증에 실패하면 실패 이유와 줄을 보여 주고 **한 번 더** 고쳐 쓰게 한다.

   줄은 구분자로 나눈 뒤 값을 감싼 괄호·따옴표를 떼고 읽는다(NWPU VHR-10의 `(563,478),(630,573),1`).

어떤 출처의 카드든 **검증을 통과해야만** 쓴다: 이미지 최대 200장의 어노테이션에 규칙을 적용해 읽을 수 없는 줄 2% 이하, 매핑되지 않은 클래스 값 2% 이하, 이미지 밖 박스(2% 여유) 5% 이하, 넓이 0인 박스 5% 이하, 박스가 하나 이상이어야 한다. 통과하지 못하면 다음 출처로 넘어가고, 모두 실패하면 *Annotations Fovea could not read*로 알리고 이미지는 라벨 없이 인덱싱한다(원본은 건드리지 않음).

외부로 나가는 것: AI 공급자에게 폴더 이름과 표본 줄(이미지는 보내지 않음), 웹 검색과 Hugging Face 검색에 짐작한 데이터셋 이름. **검증을 통과한** 규칙만 30일 동안 저장해 다시 감지해도 비용이 들지 않는다(실패한 규칙은 다음에 다시 묻는다). 이 호출은 데이터셋이 생기기 전이라 *Runs & cost*에 기록되지 않는 대신, 등록 화면과 Overview의 형식 카드에 규칙을 쓴 모델과 비용이 표시된다. 실측(Claude Opus 5.5 기본 모델): 공개 데이터셋의 웹 조회 20~40초·$0.15~0.30(Anthropic 웹 검색은 검색 1,000회당 $10 포함), 비공개 형식 10~15초·약 $0.02, 데이터셋 판별 약 $0.0002.

## 등록 화면

Add dataset 대화상자에 형식 카드가 보인다: 형식 이름과 출처(내장 / AI가 표본에서 / 공식 문서에서), 규칙을 쓴 모델과 비용(저장된 답이면 *saved answer*), 공식 링크, 검증 결과(확인한 파일 수, 박스·영역 수, 읽지 못한 줄), 클래스, 속성과 영역 이름, 그리고 **변환된 박스를 그린 표본 이미지 4장**(영역은 빗금). 입력하는 동안의 자동 감지는 AI를 쓰지 않으며, *Detect*·Enter·Browse·Register 때 AI에 묻는다(진행 중인 AI 감지는 입력 멈춤으로 시작되는 감지가 끊지 않는다). AI가 읽는 데 몇 초 이상 걸리면 최대 1분까지 걸릴 수 있다는 안내가 나온다.

## 변환

등록(또는 *Re-detect layout & rescan*) 후 인덱싱 잡이 먼저 변환한다.

- 출력 폴더: 원본 옆 `labels/`(VisDrone `…-DET-train/annotations` → `…-DET-train/labels`). 그 이름이 다른 사람의 라벨 파일로 차 있거나 원본 폴더 자체이면 `<이름>_yolo`.
- 원본 어노테이션 파일은 절대 수정하지 않는다. 출력 폴더에 Fovea가 쓰지 않은 `.txt`가 있으면 변환을 거부한다.
- 모든 분할이 **하나의 클래스 목록**을 쓴다. 카드가 클래스를 고정하면(VisDrone 1~10 → 0~9) 그대로, 데이터에 이름이 적힌 형식(KITTI·VOC·DOTA·LabelMe·CVAT)은 모든 분할의 이름을 모아 공식 순서 → 나머지 자연 정렬 순.
- 회전 박스(DOTA)와 다각형(LabelMe·CVAT)은 다각형 행(YOLO-OBB·segmentation 형식)으로 보존한다. Annotate에서 다각형 행은 클래스 변경·삭제만 된다. COCO 마스크는 박스로 바꾼다.
- 데이터셋 클래스는 사용자가 이름을 정하지 않았으면 형식의 클래스(*from format*)가 되고, 공식 정의가 있는 클래스는 빈 클래스 카드에 정의가 채워진다(직접 쓴 카드는 덮어쓰지 않음).
- 중간에 취소되면 `.fovea-format.json`의 `state`가 `converting`으로 남아 다음 인덱싱에서 다시 변환한다. 끝난 폴더는 다시 변환하지 않는다.
- **Overview → Dataset**의 형식 카드에서 *Convert again*을 누르면 원본에서 다시 변환한다. Fovea에서 고친 내용은 원본 기준으로 바뀌지만, 그 전에 라벨 버전을 저장하므로 *History → Restore*로 되돌릴 수 있다.

분할 이름: `VisDrone2019-DET-train`·`-val`·`-test-dev`처럼 공통 접두사 뒤에 분할 단어가 있으면 `train`·`val`·`test-dev`로 읽는다(내보내기는 분할을 첫 단어로 매핑하므로 예전에는 `visdrone2019-det-val`이 학습용으로 들어갔다). COCO처럼 `train2017`·`val2017` 폴더와 `annotations/instances_*.json`이 있으면 폴더마다 자기 파일을 쓰는 분할(`train`·`val`)이 된다.

## 속성과 영역

변환 폴더의 `.fovea-annotations.jsonl`에 이미지별로 각 박스의 좌표와 속성, 영역이 기록된다. 인덱스는 인덱싱과 라벨 저장 때마다 이것을 다시 읽는다.

- **박스 속성**은 박스가 변환된 좌표를 유지하는 동안 따라간다. 박스를 옮기거나 크기를 바꾸면 새 어노테이션으로 보고 속성이 사라지며, 클래스만 바꾸면 유지된다.
- **Annotate**: 선택한 박스의 속성이 HUD에 표시되고(박스 목록에서는 이름에 마우스를 올리면 보임), 영역은 빗금과 점선으로 그려지며 선택되지 않는다. 이미지 정보에 영역 수가 나온다.
- **Explore → Filters → Annotation attributes**: 속성 값별로 그 값을 가진 박스가 있는 이미지만 보기(`attr=occlusion:2`), 영역이 있는 이미지 보기(`attr=region:ignored`). 같은 속성의 값끼리는 OR, 다른 속성끼리는 AND. 옆의 숫자는 다른 필터를 적용한 이미지 수다.
- **Objects**: 같은 `attr` 필터가 이미지가 아니라 **박스 자체**를 거른다(예: 심하게 가려진 박스만).
- **AI**: 중심이 영역 안에 있는 AI 제안·감사 소견(빠진 객체)·일괄 쓰기·보간 제안은 만들지 않는다 — 객체 단위로 라벨링되지 않은 영역이라 수락하거나 검증할 대상이 없다.
- **Compare**: VisDrone·KITTI 평가 규칙처럼, 중심이 영역 안에 있는 정답 박스와 예측은 맞음·틀림 어느 쪽으로도 세지 않는다.

## 내장 형식

| 형식 | 파일 | 클래스 | 속성 | 영역 |
|---|---|---|---|---|
| VisDrone-DET | 이미지별 `.txt`, `left,top,width,height,score,category,truncation,occlusion`(픽셀) | 1~10 → pedestrian … motor | truncation(0/1), occlusion(0/1/2) | 0 ignored regions, 11 others |
| KITTI object | 이미지별 `.txt`, 공백 구분, 박스는 4~7열 `xyxy` 픽셀 | Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc | truncated(실수), occluded(0~3) | DontCare |
| DOTA | 이미지별 `.txt`, 네 꼭짓점 + 카테고리 + difficult, `imagesource:`·`gsd:` 머리줄 | 데이터의 이름(v2.0 순서 우선) | difficult | — |
| Pascal VOC | 이미지별 `.xml`(`<object><name>`, `<bndbox>`) | 데이터의 이름(VOC 20 순서 우선) | difficult, truncated, occluded, pose | — |
| LabelMe | 이미지별 `.json`(`shapes`) | 데이터의 이름 | 도형 `flags` | — |
| COCO | 하나의 `.json`(`images`·`annotations`·`categories`) | 카테고리 id 순 | 어노테이션 `attributes` | `iscrowd=1` → crowd |
| CVAT for images | 하나의 `annotations.xml`(`<image>` 안의 `<box>`·`<polygon>`) | 작업의 라벨 순 | occluded, `<attribute>` | — |

VisDrone-VID·MOT처럼 시퀀스마다 파일이 하나인 형식은 아직 지원하지 않는다(이미지별 파일과 짝이 맞지 않아 검증에서 걸러진다).
