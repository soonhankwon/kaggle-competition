from numpy.random import rand
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

heart_disease_df = pd.read_csv('./playground-series-s6e2/train.csv')
y_heart_disease_df = heart_disease_df['Heart Disease'].map(lambda x: 1 if x == 'Presence' else 0)
heart_disease_df = heart_disease_df.drop(columns=['Heart Disease', 'id'])

X_train, X_test, y_train, y_test = train_test_split(heart_disease_df, y_heart_disease_df, test_size=0.2, random_state=111)

# 결정트리, Random Forest, 로지스틱 회귀 Classifier 클래스
dt_clf = DecisionTreeClassifier(random_state=11)

# DecisionTreeClassifier 학습/예측
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)

# 모델 평가
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)
    print('오차행렬:\n', confusion)
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, ROC: {roc_auc:.4f}\n')

get_clf_eval(y_test, dt_pred)
"""
정확도: 0.8251, 정밀도: 0.8060, 재현율: 0.8057, F1: 0.8058, ROC: 0.8233
"""

parameters = {
    'max_depth': [2,3,5,10],
    'min_samples_split': [2,3,5],
    'min_samples_leaf': [1,5,8]
}

# StratifiedKFlod 사용
grid_clf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=111))
grid_clf.fit(X_train, y_train)

print('GridSearchCV 최적 하이퍼 파라미터: ', grid_clf.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_clf.best_score_))
best_clf = grid_clf.best_estimator_

# GridSearchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행
dpredictions = best_clf.predict(X_test)
get_clf_eval(y_test, dpredictions)

"""
GridSearchCV 최적 하이퍼 파라미터:  {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 2}
GridSearchCV 최고 정확도: 0.8815
오차행렬:
 [[62477  6753]
 [ 8067 48703]]
정확도: 0.8824, 정밀도: 0.8782, 재현율: 0.8579, F1: 0.8679, ROC: 0.8802
"""


# 3. Kaggle 제출용 예측 결과 CSV 생성 (gender_submission.csv 형식)
# 전체 데이터로 재학습
best_clf.fit(heart_disease_df, y_heart_disease_df)
test_df = pd.read_csv('./playground-series-s6e2/test.csv', sep=',')
test_pred = best_clf.predict(test_df.drop(columns=['id']))
submission = pd.DataFrame({
    'id': test_df['id'],
    'Heart Disease': test_pred.astype(int)
})

submission.to_csv('submission.csv', index=False)