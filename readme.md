![image](https://github.com/Liuian/predict-stock-price_independent-study_NCKU/blob/main/%E4%B8%8D%E5%88%86%E7%B3%BB%E5%B0%88%E9%A1%8C%E6%B5%B7%E5%A0%B1_page-0001.jpg)
##  實作方法總結
原始、預測資料：  
美國一個城市Ames, Iowa的真實房價資料  
train.csv: 1422 homes with 75 variables & real price(1422, 75)  
test.csv: (1459, 74) 
  
簡單的填補缺失值＋XGBoost：一開始我用簡單的方法填補缺失值，搭配XGBoost訓練模型。可以看到成效不太好，出來的成果排名在Kaggle上只有PR47而已。  

改良填補缺失值方法：為了提升預測精準度，我開始對資料的特性做完整的研究。因此可以用更具有邏輯的法填補缺失值，可以看到預測精準度大幅提升，我的排名也因此上升到PR67。  

改進XGBoost選擇參數的方法：利用Randomize CV，總共訓練模型50次，每一次的參數選擇都是隨機從我指定的幾個值裡面取的，做完以後取出結果最好的那一組作為欲使用的參數。預測精準度也因此變得更好一些。  

特徵選擇(feature selection)：我嘗試幾種不同的方法Lasso, Random forest, Correlation, Mutual info, forward selection, backward selection，選擇出欲使用的特徵後再做訓練。  

資料預處理：將原始資料做standardize, log transformation後再做訓練。  

最好的訓練結果：XGBoost + 改善填補缺失值的方法 + randomize CV


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


