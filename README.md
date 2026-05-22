# Movie Match Lab

`Recommendation.ipynb`에서 쓰인 Netflix 영화 평점 데이터를 웹에서 바로 추천에 사용할 수 있게 바꾼 Flask 앱입니다.

## 실행 방법

1. 가상환경 생성 후 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. 서버 실행

```bash
python app.py
```

3. 브라우저에서 접속

```text
http://127.0.0.1:5000
```

## 데이터셋 준비

앱은 아래 순서로 데이터를 찾습니다.

1. `artifacts/recommender_slim.pkl.gz`
2. `data/Netflix_Dataset_Movie.csv`
3. `data/Netflix_Dataset_Rating.csv`
4. 없으면 `kagglehub`로 `rishitjavia/netflix-movie-rating-dataset` 다운로드

배포용으로는 `artifacts/recommender_slim.pkl.gz`를 우선 사용합니다. 이 파일이 있으면 대용량 원본 CSV 없이도 빠르게 시작할 수 있습니다.

## Render 배포

Render 기준 권장 설정:

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn --workers 1 --threads 2 --timeout 120 app:app`
- Health Check Path: `/api/health`

루트에 `render.yaml`과 `.python-version`도 포함했습니다.

## 원본 데이터셋 준비

원본 CSV가 필요한 경우 앱은 아래 순서로 데이터를 찾습니다.

1. `data/Netflix_Dataset_Movie.csv`
2. `data/Netflix_Dataset_Rating.csv`
3. 없으면 `kagglehub`로 `rishitjavia/netflix-movie-rating-dataset` 다운로드

네트워크나 Kaggle 접근이 안 되는 환경이면 위 두 CSV를 직접 `data/` 폴더에 넣으면 됩니다.

## 추천 방식

- 사용자가 고른 영화와 비슷한 영화를 본 사용자 집합을 찾습니다.
- 그 사용자들이 높게 평가한 다른 영화들을 집계합니다.
- 평균 평점과 평가 수를 같이 반영해서 결과를 정렬합니다.

노트북의 GNN 실험을 그대로 서비스에 올리기보다, 웹에서 바로 반응 가능한 추천 파이프라인으로 재구성한 버전입니다.
