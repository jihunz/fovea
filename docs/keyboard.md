# 키보드 단축키

이 문서의 표는 각 화면이 `fovea.keys.registerKeys(scope, title, items)`로 등록한 목록에서 생성했다. 앱 안의 `?` 대화상자도 같은 목록을 읽는다(현재 화면의 묶음이 맨 위). 둘이 어긋나지 않도록 `src/tests/test_docs.py`가 각 화면을 설치해 `keys.keyGroups()`를 읽고 이 표와 한 줄씩 비교한다. 설명은 앱에 표시되는 영어 문구 그대로다. 등록 위치: `static/app.js`(Everywhere), `static/views/explore.js`(Explore grid, Inspect), `static/views/objects.js`, `static/views/annotate.js`(`KEYMAP`), `static/views/files.js`, `static/views/ai.js`, `fovea/plugins/compare/static/compare.js`(`viewer.js`의 `VIEWER_KEYS`).

## 공통 규칙

- **물리 키 기준**: 글자·숫자·기호 키는 `KeyboardEvent.code`(자판 위치)로 비교한다. 한글 입력 상태(2벌식에서 `V`가 `ㅍ`)나 Caps Lock이 켜져 있어도 같은 키로 동작하고, 대소문자는 구분하지 않으며 Shift는 `⇧`로 따로 표기한다.
- **`⌘`** 는 macOS의 Command, Windows/Linux에서는 `Ctrl`이다(앱도 그렇게 표시한다).
- **텍스트 입력 중에는 단축키가 동작하지 않는다.** 슬라이더·체크박스·선택 상자에 포커스가 있을 때는 동작한다.
- **`G`는 이동 접두키**: `G`를 누르면 *G…* 표시가 뜨고 다음 키 하나가 이동 대상이 된다. 그 두 번째 키는 화면으로 전달되지 않으므로 `G A`가 이미지를 승인하는 일은 없다. `G` 단독에는 아무것도 묶지 않는다.
- 리뷰·수락 같은 일회성 동작은 **키를 누르고 있어도 반복되지 않는다**(자동 반복 무시).

## Everywhere (`global`)

| 키 | 동작 |
| --- | --- |
| `⌘K` | Command palette: datasets, pages, images, actions |
| `?` | Keyboard shortcuts |
| `G` `H` | Go to Home |
| `G` `O` | Overview |
| `G` `E` | Explore |
| `G` `B` | Objects |
| `G` `A` | Annotate |
| `G` `I` | AI |
| `G` `F` | Files |
| `G` `C` | Compare |
| `G` `X` | Export |
| `G` `S` | Settings |
| `Esc` | Close the dialog, menu or panel on top |

## Explore grid (`explore`)

| 키 | 동작 |
| --- | --- |
| `←` `→` `↑` `↓` | Move focus |
| `PageUp` `PageDown` | Page up / down |
| `Home` `End` | First / last image |
| `Enter` | Inspect |
| `Space` | Hold: quick look |
| `A` `F` `X` | Approve / flag / exclude (focused image or selection) |
| `U` | Clear review |
| `⇧A` | Approve the unreviewed rest (keeps flagged and excluded) |
| `S` `⇧S` | Select / select range |
| `⌘A` | Select all matching images |
| Drag | Select a rectangle (start between tiles; ⇧ or ⌘ adds) |
| `Esc` | Clear selection, then the AI search, then class isolation |
| `0`–`9` | Show only one class’s boxes |
| `O` | Show / hide boxes |
| `[` `]` | Smaller / larger tiles |
| `J` `⇧J` | Next / previous image with a label problem (AI finding, label issue, sequence or shape flag) |
| `N` | Next unreviewed image |
| `E` | Edit in Annotate |
| `/` | Search filenames |
| `/` then `⇧/` | Ask AI: search in your own words (when an AI provider is set up) |
| `⌘Z` | Undo the last review or AI decision |

`⇧/`는 대부분의 자판에서 `?`와 같은 키라서 그리드에서는 도움말이 열린다. Ask AI는 `/`로 검색창에 들어간 뒤 빈 검색창에서 `⇧/`(또는 검색창 왼쪽 ✦)로 켠다. Ask AI 검색창에서 `Enter`는 한글 조합 중에는 보내지 않는다.

## Inspect (`inspect`)

| 키 | 동작 |
| --- | --- |
| `←` `→` | Previous / next image |
| `Home` `End` | First / last image |
| `A` `F` `X` | Approve / flag / exclude |
| `U` | Clear review |
| `N` | Write a note (Enter saves) |
| `Space` | Auto-play |
| `O` | Show / hide boxes |
| `L` | Show / hide labels |
| `0`–`9` | Show only one class |
| `Z` | Zoom to the selected box / fit |
| `=` `-` | Zoom in / out |
| Wheel | Zoom at the pointer (drag to pan) |
| `⇧Enter` | Accept the visible proposals (not those over a box of another class, ⇄); with none left, apply the suggested fixes |
| `Delete` | Reject the visible proposals and fixes (⌘Z brings them back) |
| `{` `}` | Proposal threshold down / up |
| `I` | Ask AI to check the labels |
| `J` `⇧J` | Next / previous image with a label problem |
| `E` | Edit in Annotate |
| `⌘Z` | Undo the last review, accept or rejection |
| `Esc` | Hide a QA flag’s details, then close |

메모 입력란에서 `⌘Enter`는 메모를 저장하고 승인한다. *suggested fixes*는 AI 라벨 감사가 남긴 수정 제안이다([ai-labeling.md](ai-labeling.md#7-라벨-감사-label-audit)). `{` `}`는 제안 표시 임계값을 0.05씩 바꾸고(감사 수정 제안은 임계값과 상관없이 보인다), `⇧Enter`·`Delete`는 화면에 보이는 것만 처리한다. 데이터 품질 플래그 칩(예: *Box jumps in a sequence*)을 누르면 어느 박스인지와 비교할 앞뒤 프레임·쌍둥이 이미지 버튼이 한 줄로 열리고, `Esc`는 먼저 그 줄을 닫은 뒤 다음 번에 Inspect를 닫는다.

## Objects (`objects`)

| 키 | 동작 |
| --- | --- |
| `←` `→` `↑` `↓` | Move focus (⇧ extends the selection) |
| `0`–`9` | Reassign focused or selected boxes to a class |
| `C` | Pick a class to reassign to |
| `Del` | Delete focused or selected boxes |
| `S` | Select / deselect the focused box |
| `⇧`+Click | Select a range |
| `⌘A` | Select all loaded boxes |
| `Esc` | Cancel a class number being typed, else clear the selection |
| `Enter` | Open the image in Annotate with the box selected |
| `Space` | Hold to see the full frame |
| `⌘Z` | Undo the last change |
| `A` | Accept the AI suggestion on the box |
| `O` | Box outline on / off |
| `[` `]` | Smaller / larger tiles |
| `I` | Details panel |
| `Home` `End` | First / last box |

숫자 키는 그 숫자로 시작하는 두 자리 클래스가 있을 때만(숫자 × 10 < 클래스 수, `0`은 제외) 두 번째 숫자를 550 ms 기다린다. 예를 들어 클래스가 13개 이상이면 `1` `2` → 클래스 12이고, 클래스가 12개면 `1`은 기다리지만 클래스 12가 없으므로 `1` `2`는 클래스 1을 적용한 뒤 `2`를 새 숫자로 받아 클래스 2로 바꾼다. 기다리는 동안 `Esc`는 입력 중인 숫자를 취소한다.

## Annotate (`annotate`)

| 키 | 동작 |
| --- | --- |
| `V` `B` `P` `W` `H` | Tools: select · box · point · click-to-box (SAM) · pan |
| `B` `B` | Box tool again: drag ⇄ 2-click (corner, then corner; zoom and pan in between) |
| `0`–`9` | Class of the selected box (else for new boxes); a 2nd digit picks 10+ |
| `[` `]` | Previous / next class |
| `/` | Search classes |
| `Del` | Delete box · reject proposal |
| `⇧Del` | Reject all visible proposals |
| `Enter` | Accept proposal |
| `⇧Enter` | Accept all visible proposals |
| `{` `}` | Proposal threshold −/+ 0.05 |
| `Tab` `⇧Tab` | Cycle proposals, then boxes |
| `I` | Propose with the current AI model |
| `⇧I` | Choose AI model |
| `R` | Refine selected box (SAM) |
| `C` | Previous frame → proposals |
| `⇧C` | Copy previous frame boxes |
| `K` | Keyframe · interpolate between keyframes |
| `` ` `` | Previous-frame ghost |
| `Alt` + `←` `↑` `→` `↓` | Move box or focused edge (⇧ ×10) |
| `E` | Focus an edge of the selected box |
| `⌘C` `⌘V` `⌘D` | Copy · paste · duplicate boxes |
| `O` | Hide boxes (hold to peek) |
| `⇧O` | Box fill 0 / 8 / 20% |
| `L` | Labels: all · selected · off |
| `T` | Crosshair |
| `M` | Loupe |
| `Z` | Zoom to box / back |
| `⇧Z` | 100% |
| `=` `-` | Zoom in / out |
| `F` | Fit |
| `←` `→` | Previous / next image |
| `Home` `End` | First / last image |
| `N` | Next image without boxes |
| `⇧N` | Next image with proposals |
| `,` `.` | Previous / next frame of this sequence |
| `\` | Sequence filmstrip |
| `⇧A` `⇧F` `⇧X` | Approve · flag · exclude (again: keep it, go on) |
| `⇧U` | Clear review |
| `⌘Z` `⌘⇧Z` | Undo · redo |
| `⌘S` | Save now |
| `Space` + drag | Pan |
| Wheel | Zoom (trackpad: scroll pans, pinch zooms) |
| `Alt` + drag | Always move, never resize |

보충:

- 첫 숫자는 바로 적용된다. 그 숫자로 시작하는 두 자리 클래스가 있으면(숫자 × 10 < 클래스 수) 600 ms 안에 같은 박스에서 누른 두 번째 숫자가 그것을 바꾼다(같은 실행 취소 단계, 예: 클래스 13개 이상에서 `1` `2` → 12).
- `B` 2-클릭 모드에서는 첫 모서리를 찍은 뒤 휠·`=`/`-` 줌과 `Space` 이동을 해도 모서리가 유지된다. `Esc`로 첫 모서리를 지운다.
- `P`(고정 크기 박스) 도구에서 `Alt`+휠은 너비, `Alt`+`⇧`+휠은 높이를 바꾼다.
- `,`/`.`는 같은 시퀀스의 프레임을 파일 이름 순으로 이동하며, 현재 목록(작업 묶음) 밖 프레임은 건너뛰고 몇 개를 건너뛰었는지 알려 준다.
- `O`를 누르고 있으면 그동안만 박스를 숨긴다(놓으면 돌아옴).
- `Tab`/`⇧Tab`은 보이는 제안을 먼저, 그다음 숨기지 않은 클래스의 박스를 차례로 선택하고 끝에서 처음으로 돈다. 포인터가 그림 위에 있으면 항상 그렇고, 그림 밖에서는 무언가 선택되어 있고 포커스가 버튼·입력란 같은 컨트롤에 있지 않을 때만 그렇다. 그 밖에는 브라우저의 기본 포커스 이동이라 키보드만으로 패널의 컨트롤에 갈 수 있다. `Tab`으로 간 컨트롤에서는 `Enter`·`Space`가 그 컨트롤을 누르고(제안 수락이 아님), 슬라이더·선택 상자에서는 `←` `→` `↑` `↓` `Home` `End`가 값을 바꾼다. 마우스로 누른 뒤 포커스만 남은 버튼에서는 `Enter`가 평소대로 제안을 수락하고 `←` `→`가 이미지를 넘긴다.
- `⌘S`는 박스와 함께 아직 서버에 닿지 않은 제안 결정(저장 상태 *Decisions not saved*)도 바로 다시 보낸다.
- `I`는 지금 이미지에 대한 요청이다. 답을 기다리는 동안 다른 이미지로 가도 되고(결과는 요청한 이미지에 저장되어 돌아가면 보임), 거기서 `I`를 누르면 그 이미지의 요청이 따로 시작된다(동시에 3장까지). 패널 상태 줄의 **Stop**은 기다림만 끝낸다.
- `K`는 이 프레임을 키프레임으로 표시한다(같은 프레임에서 다시 누르면 해제). 같은 시퀀스의 다른 프레임에서 `K`를 누르면 두 키프레임 사이 프레임마다 보간 제안을 만들고, 방금 프레임이 다음 키프레임이 된다. 같은 클래스 박스끼리, 그리고 넘어짐처럼 제자리에서 클래스가 바뀐 박스끼리 짝지으며, 클래스가 바뀌는 박스의 제안은 가까운 키프레임의 클래스를 받는다([ai-labeling.md](ai-labeling.md#3-제안-검토-annotate--explore--objects)).
- `Esc`는 한 번에 하나씩 지운다: 진행 중인 이동·크기 조절(취소) → 2-클릭의 첫 모서리 → 남아 있는 안내 줄(예: *Fill from neighbours* 결과) → `E` 한 변 이동 → 눌러 둔 데이터 품질 플래그 칩의 주황 표시 → 선택.
- 이미지 아래 데이터 품질 플래그 칩을 누르면 무엇이 다른지 주황색으로 그린다. *Box missing in one frame of a sequence* 칩 옆 **Fill from neighbours**는 앞뒤 프레임에서 빠진 박스를 보간 제안으로 만들고, `Enter`로 수락한다([data-quality.md](data-quality.md)).
- 아래 위치 입력란은 번호, `#id`, 파일 이름 일부를 받는다(`Enter` 이동, `Esc` 취소).
- `⇧N`은 대기 중인 제안이 있는 다음 이미지로 가지만, 그 제안이 모두 여기서 숨는 이미지(`{` `}` 임계값 아래, 숨긴 클래스, 같은 클래스 박스와 겹치는 *already labeled*)는 건너뛰고 몇 장을 건너뛰었는지 알려 준다. 한 번에 25장까지 확인한다. 숨은 제안은 하나씩 고를 수 없으므로 패널 *Proposals* 칸의 **Reject hidden**으로 한 번에 거절한다(임계값 아래와 *already labeled*만, `⌘Z`로 되돌림).
- `{` `}`는 패널 *Proposals* 칸의 같은 이름 버튼으로도 누를 수 있다(0.05씩, 0과 1에서 비활성). `Tab`으로 간 버튼은 임계값이 바뀐 뒤에도 포커스가 남아 `Enter`·`Space`로 계속 바꾼다.
- 패널의 *Advance after ⇧A / ⇧X*가 켜져 있으면(기본) 승인·제외 후 다음 이미지로 넘어간다. `⇧F`는 넘어가지 않는다. 이미 같은 상태인 이미지에서 다시 누르면 상태는 그대로 두고(해제하지 않음) 이 설정에 따라 다음으로 넘어간다. 해제는 `⇧U`. 대기 중인 제안이 남아 있으면 `⇧A`는 먼저 안내를 띄우고, 3초 안에 한 번 더 누르면 승인한다.

## Files (`files`)

| 키 | 동작 |
| --- | --- |
| `↑` `↓` | Move in the folder list |
| `PageUp` `PageDown` | Move 15 rows |
| `Home` `End` | First / last entry |
| `Enter` | Open the folder or preview the file |
| `Backspace` | Up one folder |

폴더 목록에 포커스가 있을 때 동작한다. 목록에 들어가려면 행을 클릭하거나, 목록 바로 위 경로 줄의 **Copy path** 버튼에서 `Tab`을 한 번 더 누른다. 페이지를 새로 열면 `Tab`은 사이드바, 데이터셋 헤더, 탭, 경로 줄을 먼저 지나므로 `Tab` 한 번으로는 들어가지 않는다(목록 전체가 탭 정지점 하나). 목록 안에서 폴더를 열거나(`Enter`, 클릭) 올라가도(`Backspace`) 포커스는 목록에 남고, 올라가면 방금 있던 폴더 행에 놓인다.

## Compare viewer (`compare`)

| 키 | 동작 |
| --- | --- |
| `←` `→` | Previous / next image |
| `[` `]` | Confidence threshold −/+ 0.05 |
| `{` `}` | Confidence threshold −/+ 0.01 |
| `E` | Errors only |
| `L` | Labels on / off |
| `B` | Show predictions below the threshold |
| `F` | Flag / unflag this image |
| `Enter` | Edit ground truth in Annotate |
| `⌘C` | Copy side-by-side PNG |
| `Esc` | Close |

텍스트를 선택한 상태에서는 `⌘C`가 PNG 대신 텍스트를 복사한다.

## AI tab (`ai`)

| 키 | 동작 |
| --- | --- |
| `↑` `↓` | Class cards: previous / next class (list focused) |

## 목록에 없는 키보드 조작

`?` 대화상자에는 나오지 않지만 표준 위젯 동작으로 쓸 수 있다.

| 위치 | 키 |
| --- | --- |
| 명령 팔레트 | `↑` `↓` 이동, `Enter` 실행, `Esc` 닫기 |
| 데이터셋 탭 막대 | `←` `→` 탭 이동, `Home` `End` 처음/끝 |
| Add dataset 폴더 선택기 | `↑` `↓` 이동, `Backspace` 상위 폴더, 경로 입력란에서 `Enter` |
| 메뉴·대화상자 | `↑` `↓` `Home` `End` 이동, `Enter` 선택, `Esc` 닫기, 대화상자 안에서 `Tab` 순환 |
| Overview 클래스 이름 | `Enter` 저장 |
| Explore 검색창 | `Esc` 지우기 |
