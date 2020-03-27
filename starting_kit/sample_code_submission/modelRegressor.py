import pickle
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor


class modelRegressor (BaseEstimator):
<<<<<<< HEAD
    def __init__(self, base_estimator, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=300, n_jobs=10, oob_score=False, verbose=False, warm_start=False):

=======
    def __init__(self, base_estimator, bootstrap, bootstrap_features, max_features, max_samples, n_estimators, n_jobs, oob_score, verbose, warm_start):
        
>>>>>>> 265004997adba0f1a9c6a2a865d2aa4aab7aa7fb
        self.base_estimator = base_estimator
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.max_features = max_features
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.verbose = verbose
        self.warm_start = warm_start
        self.reg = BaggingRegressor(base_estimator=base_estimator, bootstrap=bootstrap, bootstrap_features=bootstrap_features, max_features=max_features, max_samples=max_samples, n_estimators=n_estimators, n_jobs=n_jobs, oob_score=oob_score, verbose=verbose, warm_start=warm_start)
<<<<<<< HEAD


    def fit(self, X, y, sample_weights=None):
        self.reg.fit(X, y)
        return self


    def predict(self, X):
        y_pred = self.reg.predict(X)
        return y_pred


=======
        
        
    def fit(self, X, y, sample_weights=None):
        self.reg.fit(X, y)
        return self
    
    
    def predict(self, X):
        y_pred = self.reg.predict(X)
        return y_pred
    
    
>>>>>>> 265004997adba0f1a9c6a2a865d2aa4aab7aa7fb
    def save(self, save_directory):
        info = dict(base_estimator = self.base_estimator,
                    bootstrat = self.bootstrat,
                    bootstrap_features = self.bootstrap_features,
                    max_features = self.max_features,
                    max_samples = self.max_samples,
                    n_estimators = self.n_estimators,
<<<<<<< HEAD
                    n_jobs = self.n_jobs,
=======
                    n_jobs = self.n_jobs, 
>>>>>>> 265004997adba0f1a9c6a2a865d2aa4aab7aa7fb
                    oob_score = self.oob_score,
                    verbose = self.verbose,
                    warm_start = self.warm_start)
        info_path = os.path.join(save_directory, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f)
        return self
<<<<<<< HEAD


=======
    
    
>>>>>>> 265004997adba0f1a9c6a2a865d2aa4aab7aa7fb
    def load(self, save_directory):
        info_path = os.path.join(save_directory, 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
            self.base_estimator = info['base_estimator']
            self.bootstrat = info['bootstrat']
            self.bootstrap_features = info['bootstrap_features']
            self.max_features = info['max_features']
            self.max_samples = info['max_samples']
            self.n_estimators = info['n_estimators']
            self.n_jobs = info['n_jobs']
            self.oob_score = info['oob_score']
            self.verbose = info['verbose']
            self.warm_start = info['warm_start']
<<<<<<< HEAD
        return self
=======
        return self
>>>>>>> 265004997adba0f1a9c6a2a865d2aa4aab7aa7fb
