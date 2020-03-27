<<<<<<< HEAD
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
=======
from data_manager import DataManager
from data_io import write
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
>>>>>>> 265004997adba0f1a9c6a2a865d2aa4aab7aa7fb

class preprocess :
    def __init__(self):
        self.reg = IsolationForest(max_samples=100)
<<<<<<< HEAD

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
=======
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
>>>>>>> 265004997adba0f1a9c6a2a865d2aa4aab7aa7fb
