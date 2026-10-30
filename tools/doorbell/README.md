# 초인종 추가 학습 도구 (BL-001, BL-003, BL-006)

현재 상태: 데이터 준비·학습·평가 코드입니다. 학습 음원이 없으며, 개선 모델은 아직 학습하거나 앱에 적용하지 않았습니다.
이 도구는 앱의 모델·기준값·6종 동시 감지·마이크 개인정보 정책을 변경하지 않습니다.

## 데이터 준비

사용 권한이 확보된 외부 음원을 개발 PC에서만 준비합니다. 앱으로 주변 소리를 수집하는 기능은 추가하지 않습니다.
유튜브 링크만으로 학습하지 않으며 이 도구에는 다운로드 기능이 없습니다.

- 초인종 양성: 딩동형, 멜로디형, 전자음형, 인터폰 호출음. 유형마다 여러 제품·원본과 거리·배경 소음 조건을 포함합니다.
- 음성(other): 전화벨, 가전 알림, 대화, 음악, 아기 울음, 화재경보, 경적, 사이렌 등을 포함합니다.
- 양성 클립은 실제 초인종 구간 중심으로 잘라주세요. 클립 전체 정답이 각 프레임의 학습 정답이 되므로 긴 무음·대화 구간은 피합니다.
- WAV 형식: 16 kHz, 모노, PCM 16비트, 0.975초 이상. 평가는 길이가 비슷한 클립으로 구성합니다.

`manifest.example.csv`를 복사해 실제 파일 경로와 이용 근거(license)를 적습니다. 예시는 형식 설명용 6행이며 학습에 충분한 데이터가 아닙니다. 경로는 CSV가 있는 폴더 기준입니다.
`license`는 실제 권한/라이선스 기록이며 도구가 그 유효성을 법적으로 판정하지는 않습니다.

`source_id`는 원본 녹음/영상 ID입니다. 같은 원본을 자르거나 증강한 음원은 같은 ID와 같은 split을 사용합니다.
train/val/test 각각 양성과 음성이 필요합니다. 시험에는 학습에 사용하지 않은 제품도 포함하세요.
검사기는 중복 경로·동일 PCM·원본 ID의 split 간 중복을 거부합니다. 다른 제품인지, 수정된 중복인지까지 자동 판정하지는 않습니다.

```powershell
python tools/doorbell/train.py validate tools/doorbell/clips.csv
python -m unittest discover -s tools/doorbell -p "test_*.py"
```

## 학습과 평가

별도 ML 환경에 TensorFlow와 NumPy, 로컬 원본 YAMNet SavedModel이 필요합니다. 이 저장소에는 새 의존성을 추가하거나 자동 설치하지 않았습니다.
프로젝트 전용 ML 가상환경 경로는 `.venv-ml`입니다. 학습용 음원이 없어 실제 음원 학습·변환 실행은 아직 검증하지 못했습니다.
가상환경의 Python을 직접 지정하면 활성화 없이 실행할 수 있습니다.

```powershell
.\.venv-ml\Scripts\python.exe tools/doorbell/train.py validate tools/doorbell/clips.csv
.\.venv-ml\Scripts\python.exe tools/doorbell/train.py train tools/doorbell/clips.csv --yamnet C:/models/yamnet --output tools/doorbell/run-candidate
```

[TensorFlow 공식 전이학습 예제](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)의 YAMNet을 로컬 SavedModel로 준비합니다.
`--yamnet`은 `saved_model.pb`가 있는 디렉터리입니다. 앱의 `yamnet.tflite` 파일이 아닙니다.
원본 모델은 waveform 호출 시 scores/embeddings/spectrogram을 반환해야 합니다. 도구는 네트워크 요청을 하지 않습니다.

```powershell
python tools/doorbell/train.py train tools/doorbell/clips.csv --yamnet C:/models/yamnet --output tools/doorbell/run-candidate --max-fpr 0.05
```

1024차원 YAMNet 특징 위에 초인종 이진 분류기를 학습합니다. val에서만 기준값을 고르고 test에서 기존 Doorbell/Ding-dong 0.5 규칙과 비교합니다.
report.json에 감지율, 음성 클립 오탐률, 초인종 유형별 감지율, 데이터·모델 해시를 기록합니다.
`offline_gate_passed`는 test 감지율이 기존보다 높고, 오탐률이 기존과 지정 상한 이하인 경우만 true입니다.
SavedModel 기반 비교는 현재 앱 TFLite를 직접 실행한 결과가 아니므로 앱 기준 성능 검증을 대체하지 않습니다.
클립 최대 점수 집계는 긴 클립에서 오탐 확률이 커집니다. 클립 오탐률은 PRD의 시간당 오탐 횟수와 다릅니다.

출력은 `doorbell_head_candidate.tflite`와 `report.json`입니다. 기존 폴더를 덮어쓰지 않습니다.
변환한 head는 시험 데이터에서 Keras 출력과 수치 오차를 비교합니다. 실패한 후보도 비교 기록용으로 내보내지만 자동 배포하지 않습니다.

## 앱 적용 전 남은 작업

**이 출력은 앱의 yamnet.tflite 대체 파일이 아닙니다.** 입력은 오디오가 아닌 `[N,1024]` 특징이며 출력은 `[N,1]` 초인종 점수입니다.
현재 앱은 오디오→521개 점수 모델을 사용하므로 다음 단계에서 특징 출력이 있는 호환 모델과 Android 실행 연결이 필요합니다.
다른 5종의 기존 점수를 유지하고 초인종 판정만 검증된 head로 교체하는 방식으로 통합합니다.
기존 초인종 판정과 단순 OR 조합하면 오탐률이 달라지므로 그 조합도 따로 평가해야 합니다.

후보가 오프라인 평가를 통과한 뒤 실제 휴대전화에서 미사용 초인종 제품, 다른 5종, 생활 소음으로 재검증해야 합니다.
3초 이내 알림·시간당 오탐·배터리 목표까지 확인한 뒤 앱 모델을 변경합니다. 아직 인식률 개선을 주장할 근거는 없습니다.
