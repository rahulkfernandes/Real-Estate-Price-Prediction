import json
import pickle
import numpy as np

__LOCATIONS = None
__DATA_COLUMNS = None
__MODEL = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __DATA_COLUMNS.index(location)
    except:
        loc_index = -1
    x = np.zeros(len(__DATA_COLUMNS))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1
    
    return round(__MODEL.predict([x])[0], 2)

def get_location_names():
    return __LOCATIONS

def load_saved_artifacts():
    print("Loading Saved Artifacts")
    global __DATA_COLUMNS
    global __LOCATIONS
    global __MODEL
    with open('./artifacts/columns.json', 'r') as file:
        __DATA_COLUMNS = json.load(file)['data_columns']
        __LOCATIONS = __DATA_COLUMNS[3:]
    
    with open('./artifacts/bengaluru_home_price_model2022-12-14.pkl', 'rb') as f:
        __MODEL = pickle.load(f)
    print("Loaded Artifacts")

if __name__ == "__main__":
    load_saved_artifacts()
    #print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000,3,2))
    print(get_estimated_price('1st Phase JP Nagar',2000,2,2))
    print(get_estimated_price('Ejipura',1000,2,2))
    