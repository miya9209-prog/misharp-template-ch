# 미샵 템플릿 OS

상세페이지 자동화를 위한 이미지 편집 템플릿 제공 프로그램입니다.

## 이번 수정본 핵심
- OpenCV(cv2)가 없어도 앱이 죽지 않도록 안전 처리
- OpenCV 설치 시 더 정밀한 AI 영역 추천 활성화
- 템플릿 관리 화면을 썸네일 카드형으로 개선
- JPG + 포토샵 JSX + assets 패키지 ZIP 출력 유지

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 참고
- `opencv-python-headless`를 사용합니다.
- 직접 PSD 바이너리를 생성하는 방식이 아니라, 포토샵에서 후편집 가능한 JSX 패키지 방식입니다.
