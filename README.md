# 1. EDA

## 1-1. info
```python
<class 'pandas.DataFrame'>
RangeIndex: 630000 entries, 0 to 629999
Data columns (total 15 columns):
 #   Column                   Non-Null Count   Dtype  
---  ------                   --------------   -----  
 0   id                       630000 non-null  int64  
 1   Age                      630000 non-null  int64  
 2   Sex                      630000 non-null  int64  
 3   Chest pain type          630000 non-null  int64  
 4   BP                       630000 non-null  int64  
 5   Cholesterol              630000 non-null  int64  
 6   FBS over 120             630000 non-null  int64  
 7   EKG results              630000 non-null  int64  
 8   Max HR                   630000 non-null  int64  
 9   Exercise angina          630000 non-null  int64  
 10  ST depression            630000 non-null  float64
 11  Slope of ST              630000 non-null  int64  
 12  Number of vessels fluro  630000 non-null  int64  
 13  Thallium                 630000 non-null  int64  
 14  Heart Disease            630000 non-null  str    
dtypes: float64(1), int64(13), str(1)
memory usage: 72.1 MB
None
```

- id: 식별자
- Age: 나이
- Sex: 성별
- Chest pain type: 가슴통증 타입
- BP: Blood Pressure
- Cholesterol: 콜레스트롤
- FBS over 120: 공복혈당 120mg/dL 이상
- EKG results: 심전도
- Max HR: 최대 심박수
- Exercise angina: 운동 유발성 협심증(운동 등 많은 산소를 필요로 하는 활동시 가슴통증)
- ST depression: ST 분절 하강
- Slope of ST: ST 분절 기울기
- Number of vessels fluro: 투시 조영술
- Thallium: 탈륨(독성이 강한 원소)
- Heart Disease: 라벨 데이터

# 2. ML 모델 (심장병 예측)

## Decision Tree

- **알고리즘**: `DecisionTreeClassifier`
- **하이퍼파라미터 튜닝**: GridSearchCV + StratifiedKFold(5-fold)
- **최적 파라미터**: max_depth=10, min_samples_leaf=8, min_samples_split=2
- **성능**: 정확도 88.2%, ROC-AUC 88.0%
- **Kaggle Public Score**: 0.87626
- **학습**: 전체 학습 데이터로 최종 재학습 후 test 데이터 예측