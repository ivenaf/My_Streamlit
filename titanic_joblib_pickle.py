#import previously trained classifier clf, saved as model , with joblib
import joblib
joblib.load('model')

import pickle
#import previously trained classifier clf, saved as model , with pickle
loaded_model=pickle.load(open('model', 'rb'))