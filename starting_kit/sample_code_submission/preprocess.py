from data_manager import DataManager
from data_io import write
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

class preprocess :
    def __init__(self):
        self.reg = IsolationForest(max_samples=100)
        self.scaler = StandardScaler()
    
    def fit_transform(self, X):
        X = self.scaler.fit_transform(X)
        #Feature selection
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X_train = sel.fit_transform(X)
        return X_train
        
    def transform(self, X):
        X = self.scaler.transform(X)
        #Outliers Detection
        self.reg.fit(X)
        
        #PCA
        pca = PCA(n_components = 6)
        pca.fit(X)
        N_X = pca.transform(X)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)
        return N_X
