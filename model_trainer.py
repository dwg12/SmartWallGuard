import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import joblib

# 1. 시나리오별 합성 데이터 생성 함수
def generate_synthetic_data(samples_per_class=500):
    data = []
    
    for _ in range(samples_per_class):
        # [정상 상황] 낮은 온도, 낮은 충격, 짧은 체류
        data.append([np.random.normal(24, 1), np.random.normal(16384, 200), np.random.uniform(0, 5), 0])
        
        # [배회 상황] 사람 체온, 낮은 충격, 긴 체류 시간 (30초 이상)
        data.append([np.random.normal(33, 1.5), np.random.normal(16384, 300), np.random.uniform(30, 120), 1])
        
        # [이상 충격 상황] 사람 체온, 높은 충격(담 넘기), 짧은 체류(빠른 이동)
        data.append([np.random.normal(34, 1), np.random.normal(24000, 1500), np.random.uniform(1, 10), 2])
        
        # [낙상 상황] 사람 체온, 매우 높은 충격(바닥 충돌), 중간 체류(쓰러진 채 정지)
        data.append([np.random.normal(32, 2), np.random.normal(30000, 2500), np.random.uniform(10, 20), 3])
        
        # [동물 감지] 낮은 체온(털에 의한 단열효과), 중간 충격(빠른 움직임), 매우 짧은 체류
        data.append([np.random.normal(28, 1), np.random.normal(18000, 1000), np.random.uniform(0, 3), 4])

    columns = ['avg_temp', 'max_impact', 'stay_time', 'label']
    return pd.DataFrame(data, columns=columns)

# 2. 데이터 생성 및 전처리
print("🚀 합성 데이터 생성 중...")
df = generate_synthetic_data()

X = df[['avg_temp', 'max_impact', 'stay_time']]
y = df['label']

# 학습용/테스트용 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Random Forest 모델 학습
print("🧠 AI 모델 학습 시작 (Random Forest)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. 성능 검증 (F1-score 확인)
y_pred = rf_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')

print("-" * 30)
print(f"✅ 모델 학습 완료!")
print(f"📊 F1-Score: {f1:.4f}") # 80% 이상인지 확인
print("-" * 30)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Loitering', 'Impact', 'Fall', 'Animal']))

# 5. 모델 저장
model_filename = 'model_rf.pkl'
joblib.dump(rf_model, model_filename)
print(f"💾 모델 파일 저장 완료: {model_filename}")