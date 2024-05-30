![image](https://github.com/Liuian/predict-stock-price_independent-study_NCKU/blob/main/%E4%B8%8D%E5%88%86%E7%B3%BB%E5%B0%88%E9%A1%8C%E6%B5%B7%E5%A0%B1_page-0001.jpg)

## 實作方法總結
原始資料：從investing.com抓csv檔；元大0050；Period: January 1, 2014, to November 4, 2023。

1. Goldencross:
   - Moving Average: Short = 5, Long = 20
   - Total Return Rate of Backtesting Analysis: 62.48%  

2. Machine Learning-random forest:
   
   - 80% of the data is the training set, and 20% is the test set
   - R2 Score: 0.9997980640298204
   - Issue: No concept of time series  

3. LSTM:

   - Using data from the past ten days as the training set for each training.
   - The first 80% of the data is the training set.
   - R2 Score: 0.9535430373937358  

4. 綜合比較  

   - 方法：Golden cross 以實際股市走勢回測、Random forest及LSTM以預測後的股價進行回測
   - Dataset日期區間：2014.01.02~2023.11.03
   - 預測區間：2021.11.10~2023-11-03
   - Golden Cress 長均線、短均線：5, 20  

5. 結論：以下幾種情況都會影響交易的回報率


   - 不同時間區間
   - 不可以看到未來的資料random forest
   - 嘗試優化參數
   - Golden cross > random forest > random forest  



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


