# 시각 피로 최소화 원칙

Fovea는 한 사람이 수천 장의 이미지를 몇 시간씩 검수·라벨링하는 도구다. 목표는 **사람의 수고와 피로를 줄이면서도 빠르게 일할 수 있게** 하는 것이다. 이 문서는 화면의 모든 시각 효과가 따르는 규칙과 그 근거를 정리한다. 근거는 2024–2026년 논문을 우선했고, 그 기간에 해당 주제를 다룬 연구가 없을 때만 오래된 기초 연구를 연도와 함께 표시했다. 새 기능과 플러그인도 이 규칙을 따른다.

## 규칙

### 1. 이미지 전환은 즉시, 밝기 변화는 한 번

현재 이미지는 다음 이미지가 **완전히 디코딩될 때까지** 화면에 남고, 준비되면 이미지·박스·패널이 **같은 프레임에** 바뀐다. 페이드, 빈 화면, 애니메이션 자리표시자는 없다. 이웃 프레임은 미리 디코딩해 두므로 보통 대기 자체가 없다(키 입력에서 교체까지 5–10 ms).

- 연속 영상 프레임은 서로 거의 같다. 즉시 교체하면 화면의 밝기 변화가 거의 없지만, 어둡게 사라졌다 나타나는 페이드는 매 키 입력마다 어두움→밝음 한 주기를 만든다.
- 이미지 사이에 80 ms짜리 빈 화면만 끼워도 두 장면의 차이를 찾는 데 0.9초가 아니라 4.7–10.9초가 걸렸다. 빈 화면이 "무엇이 바뀌었는지" 알려 주는 움직임 신호를 지운다 [Rensink 1997]. 반대로 변화가 서서히 일어나면 큰 변화도 대부분 알아채지 못한다 [Frey 2024]. 프레임 사이 차이를 보는 검수 작업에 페이드는 피로와 정확도 모두에 불리하다.
- 밝기 도약이 클수록, 그리고 직전 밝기에 오래 적응했을수록 불편감이 커졌다 [Zuena & Pytlarz 2024].
- 깜박임이 보이는 정도는 깜박이는 영역의 크기와 밝기에 따라 커진다 [Cai et al. 2024]. 초당 몇 장씩 넘길 때 화면 전체가 어두워졌다 밝아지면 큰 면적의 깜박임이 되고, 초당 3회를 넘으면 WCAG 2.2의 번쩍임 기준(2.3.1)에도 걸린다.

구현: `static/lib/media.js`의 `createFrameCache`·`decodedImage` — Inspect, Annotate, Compare 뷰어가 사용한다. 갤러리 타일도 디코딩이 끝난 뒤 한 번에 나타난다.

### 2. 반복되거나 주기적인 밝기 변화 금지

깜박이는 상태 점, 번쩍이는 스켈레톤, 맥박처럼 밝아졌다 어두워지는 강조를 쓰지 않는다. 상태는 정적인 색과 글자로 보여 준다.

- 같은 자극에 반복 노출되면 불편감이 누적되고 기준 동공 크기가 줄었다. 적응이 아니라 누적 피로에 가깝다 [Meidan & Bonneh 2026].
- 일부 사람은 시간적 밝기 변조에 훨씬 민감하다 [Veitch & Miller 2024]. 한 사람의 강한 불편 호소는 예외가 아니라 예상된 일이다.

### 3. 시야 주변의 움직임 최소화

- 스피너는 **400 ms가 지나야** 보인다. 빨리 끝나는 작업은 아무것도 보이지 않는다. 토스트·모달·메뉴·오버레이는 튀어나오거나 미끄러지지 않고 그냥 나타난다.
- 검수처럼 빠르게 반복하는 단건 작업은 토스트 대신 **그 자리에서** 확인해 준다. 타일 배지나 Inspect 상단의 고정 상태 줄을 쓴다. 토스트는 일괄 작업과 오류에만 쓴다.
- 움직임이 시작되는 순간은 주의를 강제로 끌어간다 [Smith & Abrams 2018]. 계속되는 움직임 신호는 본 작업을 방해하고, 변화가 있을 때만 나타나는 신호는 덜 방해했다 [Qiu et al. 2026].
- OS의 "동작 줄이기" 설정이 켜져 있으면 남은 작은 전환(스위치, 색 전환)도 없앤다. 페이드는 WCAG 2.3.3과 `prefers-reduced-motion`의 대상이 아니므로, Fovea는 이 설정과 무관하게 **항상** 페이드를 쓰지 않는다.

### 4. 긴 작업에는 경과 시간

색인·복사·자동 라벨·평가처럼 오래 걸리는 작업은 경과 시간(`0:42`)을 보여 준다. 시간 표시가 없으면 기다림이 더 길고 불확실하게 느껴졌고, 남은 시간 카운트다운은 경과 시간보다 더 짜증스러웠다 [Tan & Nov 2026]. 구현: `ui.elapsedClock()`.

### 5. 밝기와 이미지 주변 배경

설정 → Visual comfort에서 기기별로 조정한다. 방과 모니터에 따라 적절한 값이 다르기 때문이다.

- **화면 어둡게(0–60%)**: Fovea가 그리는 모든 것을 균일하게 어둡게 한다. 화면 밝기를 낮춘 집단은 한 달 뒤 눈 피로 점수가 유의하게 줄었고(5점 척도에서 −0.82, P = 0.0007), 색온도("블루라이트") 소프트웨어 집단은 줄지 않았다 [Massa et al. 2025]. 읽기 편안함은 모니터 휘도가 적당할 때 가장 높았고, 더 밝게 하면 오히려 떨어졌다 [Daneels et al. 2024]. 어두운 방에서 밝은 화면을 보는 조합이 눈물막·깜박임·피로 모두에서 가장 나빴다 [Lin et al. 2026].
- **이미지 주변 배경(검정 / 짙은 회색 / 회색)**: 기본값을 거의 검정에서 **짙은 회색**으로 바꿨다. 중간 밝기 이미지와의 휘도 비가 약 67:1에서 약 15:1로 줄어든다. "회색"은 약 4:1로, 모니터와 주변 휘도의 비가 약 4:1일 때 가장 편안했다는 결과에 가깝다 [Daneels et al. 2024]. 다만 이 연구들은 방 조명과 화면의 관계를 본 것이고, 화면 안 이미지 주변 배경을 직접 비교한 2024–2026년 연구는 찾지 못했다. 그래서 기본값은 보수적으로 짙은 회색으로 두고, 나머지는 선택지로 남겼다.

### 6. 흐린 글자도 읽을 수 있게

가장 흐린 텍스트 단계(`--text-3`)도 주요 배경 위에서 4.5:1 이상이다(이전 3.4–3.6:1). 어두운 배경에서 대비가 낮은 글자색이 가장 큰 시각 피로를 만들었다 [Fan et al. 2024]. 빨간색 글자는 짧은 경고에만 쓴다.

클래스 색 위의 글자(박스 라벨 칩)는 `colors.textOn(hex)`로 검정·흰색 중 대비가 큰 쪽을 고른다. 기본 팔레트의 모든 색은 그중 하나와 4.8:1 이상이 되도록 골랐고, 숫자 키로 고르는 0–9번은 CIEDE2000 거리 20 이상 떨어져 있다. 설정 → Visual comfort → *Class colours*의 **Colour-blind safe**는 Okabe–Ito 순서(적록·청황 색각 이상에서도 구분됨)를 쓴다.

### 7. 라이트·다크 테마는 동등

어느 쪽이 모든 사람에게 덜 피곤한지는 정해지지 않았다. 사람마다 유리한 극성이 달랐고, 선호하는 모드와 실제로 더 빠른 모드가 자주 달랐다 [While & Sarvghad 2024]. 온라인 표본에서는 라이트 모드의 인지 점수가 더 높았다 [Gazit et al. 2025]. 그래서 기본값은 시스템 설정을 따르고, 한 번에 전환할 수 있게 둔다.

### 8. 휴식 알림은 선택 사항

설정에서 30·45·60분을 고를 수 있고 기본은 꺼짐이다. 실제로 입력이 있었던 시간만 세고, 5분 동안 입력이 없으면 휴식한 것으로 본다. 알림은 대화상자나 소리, 움직임 없이 조용히 표시되며, 몇 초가 아니라 1–2분 동안 먼 곳을 보도록 권한다.

- 소프트웨어 휴식 알림을 쓰는 동안 휴식이 늘고 증상이 줄었다 [Talens-Estarelles et al. 2023].
- 40분짜리 과제 안의 20초 휴식은 증상을 바꾸지 못했다 [Johnson & Rosenfield 2023].
- 부하가 높은 영상 판독에서는 시간이 지날수록 속도를 높이는 대신 적중률이 떨어졌고, 저자들은 30–40분 단위를 제안했다 [Buser et al. 2023].

근거가 엇갈리므로 기본으로 켜지 않는다.

### 9. 클릭 대상은 움직이지 않는다

나타났다 사라지는 줄이 아래 내용을 밀어내면, 사람이 겨냥한 버튼이 클릭 순간 다른 것으로 바뀐다. 실제로 잡 배너가 사라지며 화면이 약 38 px 올라가 탭을 누른 클릭이 빈 곳에 떨어졌다. 밀려나는 움직임 자체도 주의를 끌어간다(3번). 그래서 상태가 바뀌어도 레이아웃은 그대로 둔다.

- **잡 진행**: 데이터셋 헤더의 경로 줄(28 px) 안에 고정 높이(26 px) 상태 칸이 있고, 실행 중인 잡은 그 칸에 진행률·경과 시간·Cancel로 보인다. 잡이 끝나면 4초 동안 정적인 한 줄 결과(*Index finished · 7,808 images*, *Data quality checked*)를 남기고 애니메이션 없이 지워진다. 경로는 이때 끝부분만 보인다.
- **알림 줄**: *Partial index*나 *Files not found* 같은 알림 줄은 탭을 바꾸는 등 안전한 순간에만 나타나거나 사라진다. 그 사이 새로 생긴 문제는 상태 칸에서 기다리고, 해결된 문제는 *Index complete* 줄로 남았다가 다음 탭 이동 때 사라진다.
- **Annotate의 AI 패널**: 모델 선택(40 px), Propose/Batch 줄, 고정 높이(52 px) 상태 줄, 임계값 순서로 위치가 고정되어 있다. 상태 줄은 항상 무언가를 보여 주고(대기 안내, 비활성 이유, 진행, 결과, 오류, 다른 이미지에서 아직 도는 요청의 *Proposing for <파일>* 줄) 길면 잘라서 전체 문장은 툴팁에 둔다. 높이가 바뀌는 항목(노력 수준, 쓰기 대상 안내, 클래스 칩, 잡)은 모두 그 아래에 있다. Batch 대화상자의 예상 비용 칸도 최소 높이를 유지해 **Start** 버튼이 추정이 도착할 때 움직이지 않는다.
- **Explore**: 필터나 정렬을 바꾸면 첫 페이지와 보이는 썸네일이 디코딩될 때까지(최대 300 ms) 이전 그리드를 유지했다가 한 번에 바꾼다. 리뷰나 편집으로 필터에서 벗어난 타일은 사라지지 않고 제자리에서 *No longer matches*로 흐려지며, 개수는 *N changed · refresh*로 알려 준다. 필터 옆 숫자는 제자리에서 갱신되고, 아직 이전 필터 기준이면 흐리게 보인다. Ask AI 입력창은 이웃 위로 넓어지므로 툴바의 다른 버튼이 움직이지 않는다.
- **Annotate의 filmstrip**: 그림 위에 겹치는 띠라서 캔버스 크기가 바뀌지 않는다. 화면 맞춤(fit)은 띠 높이만큼을 비워 두므로 `\`로 켜고 꺼도 그림이 제자리에 있다.

### 10. AI 제안·고스트·filmstrip은 정적으로, 같은 프레임에

AI 결과는 사람이 판단할 재료다. 시선을 끌어당기거나 반복해서 변하면 안 된다(2번, 3번). 점선 제안과 신뢰도 슬라이더는 Roboflow 같은 도구의 관행이다. 돋보기, 이전 프레임 고스트(onion-skin) 같은 도구는 검출 라벨링에서 2023–2026년에 통제 실험으로 검증된 것이 없어 조용한 선택형 토글로 두었다.

- **제안**: 클래스 색 1.5 px 점선과 어두운 바탕의 신뢰도 칩. 선택하면 흰색·클래스 색 이중 점선이 된다. 깜박임, 움직이는 점선(marching ants), 페이드, 맥박 강조가 없다. 제안은 이미지와 박스와 **같은 프레임에** 나타나고 사라진다. 수락하면 그 자리에서 실선 박스로 바뀐다. 결정 결과는 토스트가 아니라 패널의 행(✓/✕)과 상태 줄에 보인다. 대부분을 수정 없이 수락할 때의 안내도 모달이 아닌 정적인 한 줄이다.
- **감사 의심**: 같은 점선에 기호(+ 빠짐, ⤢ 느슨함, ⇄ 클래스, ✕ 근거 없음)를 붙인다. 느슨한 박스 제안을 선택하면 원래 박스가 가는 흰 점선으로 함께 보인다. Objects는 제안 박스를 상세 패널과 `Space` 미리보기에 정적인 점선으로 그린다.
- **이전 프레임 고스트**(`` ` ``): 이전 프레임의 박스를 어두운 2 px 점과 밝은 1 px 점이 번갈아 나오는 정적인 점선으로 그린다. 회색 점선 한 줄은 밝은 프레임에서 보이지 않았기 때문에 두 색을 엇갈려 겹쳤고, 그래서 밝은 벽과 어두운 바닥 모두에서 보인다. 클래스 색을 쓰지 않아 현재 박스와 헷갈리지 않는다. 그림과 같은 프레임에 바뀌고, 켜고 끈 상태는 기억된다.
- **데이터 품질 플래그 고스트**: Annotate 이미지 아래나 Inspect의 플래그 칩을 누르면 무엇이 다른지를 클래스 색 대신 한 가지 주황색(`#f59e0b`)으로 그린다. 플래그가 가리키는 이 이미지의 박스는 주황 테두리, 쌍둥이·비슷한 이미지·앞 프레임의 박스나 이웃 프레임으로 본 예상 위치는 주황 점선이다. Annotate는 이 점선을 어두운 굵은 점선 위에 겹쳐 밝은 프레임에서도 보이게 하고, 다른 이미지의 박스 중 이 이미지에도 있는 것(같은 클래스, IoU ≥ 0.5)은 흐린 가는 점선으로 물러나 있게 한다. 깜박임이나 움직임 없이 그림과 같은 프레임에 나타나고, 칩을 다시 누르거나 `Esc`를 누르거나 다른 이미지로 가면 지워진다. Annotate HUD에 범례(무엇인지와 *Esc clears*)가 붙는다.
- **Filmstrip**(`\`): 높이 84 px의 띠. 썸네일은 **디코딩이 끝난 뒤에만** 보이고, 다음 시퀀스로 넘어갈 때는 그 띠에서 보일 썸네일을 그림을 바꾸기 전에 디코딩한다(최대 250 ms, 보통 미리 받아 둠). 그래서 새 그림과 새 띠가 한 프레임에 함께 바뀐다. 현재 프레임으로 점프할 때는 부드러운 스크롤 없이 즉시 이동해 현재 프레임이 가장자리에서 3분의 1 지점에 오도록 한다. 작업 묶음 밖의 프레임은 숨기지 않고 흐리게 둔다(띠의 구조가 바뀌지 않음). 그림 영역은 두 테마 모두 어둡기 때문에 띠는 고정된 어두운 색이다.

## 채택하지 않은 것

- **블루라이트·색온도 필터**: 눈 피로를 줄이지 못했다 [Massa et al. 2025; Singh et al. 2023, Cochrane].
- **다크 모드 강제**: 근거가 엇갈린다(7번 참고).
- **크로스페이드**: 두 이미지를 겹쳐 서서히 바꾸면 변화 신호가 사라지고(1번 참고), 겹치는 동안 밝기가 흔들린다.

## 새 기능·플러그인 체크리스트

- 이미지를 바꿀 때는 `fovea.media.createFrameCache()`로 디코딩된 이미지를 받아 **한 번에** 교체한다. 기존 이미지를 먼저 지우지 않는다.
- `animation`·`@keyframes`를 추가하지 않는다(스피너만 예외). 색이나 테두리가 바뀌는 짧은 hover 전환은 괜찮다.
- 기다림 표시는 `ui.spinner()`(자동 400 ms 지연)를 쓰고, 10초 이상 걸리는 작업에는 `ui.elapsedClock()`을 붙인다.
- 단건 작업의 결과는 사용자가 보고 있는 자리에서 보여 준다. 토스트는 일괄 작업과 오류에만 쓴다.
- 텍스트는 배경 대비 4.5:1 이상을 유지한다. 클래스 색 위 글자는 `colors.textOn(hex)`.
- 나타났다 사라지는 상태·경고 줄은 고정 높이 칸에 넣는다. 버튼 위쪽의 높이가 내용에 따라 바뀌지 않게 한다.
- AI 결과·참고 박스는 점선으로, 애니메이션 없이, 이미지와 같은 프레임에 그린다.
- 썸네일 띠·그리드도 디코딩된 이미지만 보여 주고, 자동 스크롤은 즉시 이동한다(부드러운 스크롤 없음).
- 단축키는 `fovea.keys`로 물리 키를 비교하고 `registerKeys`로 `?` 도움말에 등록한다([keyboard.md](keyboard.md)).

## 근거 문헌

| 문헌 | 핵심 결과 | Fovea 적용 |
|---|---|---|
| Rensink, O'Regan & Clark. *To See or Not to See: The Need for Attention to Perceive Changes in Scenes.* Psychological Science 8(5), 1997 — 기초 연구 | 240 ms 이미지 사이 80 ms 빈 화면: 변화 발견 4.7–10.9초 vs 빈 화면 없이 0.9초 | 1 |
| Frey et al. *A novel, semi-automatic procedure for generating slow change blindness stimuli.* Neuroscience of Consciousness, 2024. [doi:10.1093/nc/niae004](https://doi.org/10.1093/nc/niae004) | 서서히 일어나는 변화는 대부분 알아채지 못함 | 1 |
| Zuena & Pytlarz. *The Impact of Adaptation Time in High Dynamic Range Luminance Transitions.* J. Perceptual Imaging 7, 2024. [doi:10.2352/J.Percept.Imaging.2024.7.000401](https://doi.org/10.2352/J.Percept.Imaging.2024.7.000401) | 밝기 도약 크기와 직전 적응 시간이 클수록 불편감 증가 | 1, 5 |
| Cai et al. *elaTCSF: A Temporal Contrast Sensitivity Function for Flicker Detection and Modeling Variable Refresh Rate Flicker.* SIGGRAPH Asia 2024. [doi:10.1145/3680528.3687586](https://doi.org/10.1145/3680528.3687586) | 깜박임 가시성은 면적·휘도·이심률에 따라 달라짐 | 1, 2 |
| Meidan & Bonneh. *Pattern-induced visual discomfort and its cumulative effects revealed by pupillary measures.* Front. Hum. Neurosci., 2026. [doi:10.3389/fnhum.2025.1723675](https://doi.org/10.3389/fnhum.2025.1723675) | 반복 노출로 불편감 누적, 기준 동공 감소 | 2 |
| Veitch & Miller. *Effects of Temporal Light Modulation on Individuals Sensitive to Pattern Glare.* LEUKOS 20(3), 2024. [doi:10.1080/15502724.2023.2299210](https://doi.org/10.1080/15502724.2023.2299210) | 민감한 하위 집단은 시간적 밝기 변조에 다르게 반응 | 2 |
| Smith & Abrams. *Motion onset really does capture attention.* Atten. Percept. Psychophys. 80, 2018 — 기초 연구. [doi:10.3758/s13414-018-1548-1](https://doi.org/10.3758/s13414-018-1548-1) | 움직임 시작은 주의를 끌어감 | 3 |
| Qiu et al. *Acceleration or velocity? Exploring minimally disruptive visual motion cues for reducing motion sickness in passenger VR.* Applied Ergonomics, 2026. [doi:10.1016/j.apergo.2026.104778](https://doi.org/10.1016/j.apergo.2026.104778) | 지속적 움직임 신호는 방해, 변화 시점에만 나타나는 신호가 덜 방해 | 3 |
| Tan & Nov. *Counting the Wait: Effects of Temporal Feedback on Downstream Task Performance and Perceived Wait-Time Experience during System-Imposed Delays.* CHI 2026. [arXiv:2602.04138](https://arxiv.org/abs/2602.04138) | 시간 표시 없음 → 더 길고 불확실하게 느낌, 카운트다운 → 경과 시간보다 짜증 증가 (n=425) | 4 |
| Massa et al. *Effects of digital screen property modification on symptoms of digital eye strain.* Digital J. Ophthalmology 31(3), 2025. [doi:10.5693/djo/01.2024.04.001](https://doi.org/10.5693/djo/01.2024.04.001) | 밝기 낮춤: 눈 피로 −0.82 (P = 0.0007), 색온도 소프트웨어: 효과 없음 | 5 |
| Daneels et al. *Reading comfort in relation to monitor and ambient luminance levels.* Lighting Res. Technol., 2024. [doi:10.1177/14771535241269709](https://doi.org/10.1177/14771535241269709) | 모니터 260 cd/m², 벽 68 cd/m²에서 가장 편안, 700 cd/m²는 오히려 저하 | 5 |
| Lin et al. *Effects of ambient illuminance and mobile phone screen brightness on tear film stability, visual fatigue, and blink patterns during reading.* Cont. Lens Anterior Eye 49(1), 2026. [doi:10.1016/j.clae.2025.102515](https://doi.org/10.1016/j.clae.2025.102515) | 어두운 방 + 밝은 화면이 눈물막·깜박임·피로에서 가장 나쁨 | 5 |
| Fan et al. *The Effect of Ambient Illumination and Text Color on Visual Fatigue under Negative Polarity.* Sensors 24(11), 2024. [doi:10.3390/s24113516](https://doi.org/10.3390/s24113516) | 어두운 배경에서 빨간 글자가 가장 피로, 노랑·흰색이 가장 덜함, 주변 조명이 피로 감소 | 6 |
| While & Sarvghad. *Dark Mode or Light Mode? Exploring the Impact of Contrast Polarity on Visualization Performance Between Age Groups.* IEEE VIS 2024. [arXiv:2409.10841](https://arxiv.org/abs/2409.10841) | 유리한 극성이 사람마다 다름, 선호와 성과 불일치 | 7 |
| Gazit et al. *The dark side of the interface: examining the influence of different background modes on cognitive performance.* Ergonomics, 2025. [doi:10.1080/00140139.2025.2483451](https://doi.org/10.1080/00140139.2025.2483451) | 라이트 모드에서 인지 점수 더 높음 (n=173) | 7 |
| Talens-Estarelles et al. *The effects of breaks on digital eye strain, dry eye and binocular vision: Testing the 20-20-20 rule.* Cont. Lens Anterior Eye 46(2), 2023. [doi:10.1016/j.clae.2022.101744](https://doi.org/10.1016/j.clae.2022.101744) | 소프트웨어 휴식 알림을 쓰는 동안 휴식 증가·증상 감소 | 8 |
| Johnson & Rosenfield. *20-20-20 Rule: Are These Numbers Justified?* Optom. Vis. Sci. 100(1), 2023. [doi:10.1097/OPX.0000000000001971](https://doi.org/10.1097/OPX.0000000000001971) | 40분 과제 중 20초 휴식은 증상에 차이 없음 | 8 |
| Buser et al. *Time on task and task load in visual inspection: A four-month field study with X-ray baggage screeners.* Applied Ergonomics 111, 2023. [doi:10.1016/j.apergo.2023.103995](https://doi.org/10.1016/j.apergo.2023.103995) | 고부하에서 시간이 지날수록 속도↑·적중률↓, 30–40분 단위 제안 | 8 |
| Singh et al. *Blue-light filtering spectacle lenses for visual performance, sleep, and macular health in adults.* Cochrane Database Syst. Rev., 2023. [doi:10.1002/14651858.CD013244.pub2](https://doi.org/10.1002/14651858.CD013244.pub2) | 단기 눈 피로 감소 근거 없음 (낮은 확실성) | 채택하지 않음 |
| W3C. *WCAG 2.2*, SC 2.3.1 Three Flashes or Below Threshold, SC 2.3.3 Animation from Interactions | 초당 3회 이하 번쩍임, 페이드는 2.3.3 대상 아님 | 1, 3 |
