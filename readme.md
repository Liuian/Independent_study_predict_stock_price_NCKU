![image](https://github.com/Liuian/predict-stock-price_independent-study_NCKU/blob/main/%E4%B8%8D%E5%88%86%E7%B3%BB%E5%B0%88%E9%A1%8C%E6%B5%B7%E5%A0%B1_page-0001.jpg)



## coding紀錄-error
標準化資料
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_Data = scaler.fit_transform(Data)
...
反標準化預測結果
trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))

ValueError: non-broadcastable output operand with shape (1913,1) doesn't match the broadcast shape(1913,6)
因為scaler 同時紀錄了6個feature的轉換

改成 
from sklearn.preprocessing import MinMaxScaler
假設Data是包含六個特徵的資料框
features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
X = Data[features]
初始化MinMaxScaler
scalers = {}
對每個特徵獨立進行縮放
for feature in features:
    scaler = MinMaxScaler(feature_range=(0, 1))
    X[feature] = scaler.fit_transform(X[[feature]])
    scalers[feature] = scaler
X現在包含獨立縮放後的每個特徵的值
X_array = X.values


