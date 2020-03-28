from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class preprocess :
    def __init__(self):
        self.reg = IsolationForest(max_samples=100)

        
    def fit_transform (self, X) :
        #Outliers Detection
        self.reg.fit(X)
        self.reg.predict(X)
        
        #Feature selection
        sel = VarianceThreshold(threshold=(.7 * (1 - .7)))
        X_train= sel.fit_transform(X)

        #scaling
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X_train)

        return scaled_X
