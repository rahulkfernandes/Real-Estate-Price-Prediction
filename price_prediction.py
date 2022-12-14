import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20,10)

def convert_sqft_to_num(value):
    tokens = value.split('-')
    if len(tokens)==2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(value)
    except:
        return None

def load_data(path):
    # Data Cleaning
    df = pd.read_csv(path)
    df.drop(['area_type','society','balcony','availability'], axis='columns', inplace=True)
    df.dropna(inplace=True)
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    return df

def rm_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean = np.mean(subdf['price_per_sqft'])
        std = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft']>(mean-std)) & (subdf['price_per_sqft']<=(mean+std))]
        df_out = pd.concat([df_out, reduced_df],ignore_index=True)
    return df_out

def rm_bhk_outlisers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df['price_per_sqft']<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

def plot_hist(column, xlabel):
    plt.hist(column, rwidth=0.8)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.show()

def plot_scatter_plot(df, location):
    bhk2 = df[(df['location']==location) & (df['bhk']==2)]
    bhk3 = df[(df['location']==location) & (df['bhk']==3)]
    #matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2['total_sqft'],bhk2['price'], color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3['total_sqft'],bhk3['price'],marker='+',color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    plt.show()
    
def feature_engg(df):
    df['price_per_sqft'] = df['price']*100000/df['total_sqft']
    
    # To Reduce Dimensions
    df['location'] = df['location'].apply(lambda x: x.strip())
    location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending=False)
    df['location'] = df['location'].apply(lambda x: 'other' if x in location_stats[location_stats<=10] else x)
    
    # Outlier Removal
    df = df[~(df['total_sqft']/df['bhk']<300)] # 300sqft is min size of bedroom 
    df = rm_pps_outliers(df)
    #plot_scatter_plot(df, "Hebbal")
    df = rm_bhk_outlisers(df)
    df = df[df['bath']<df['bhk']+2]
    plot_hist(df['price_per_sqft'], 'Price Per Square Feet')
    df.drop(['size','price_per_sqft'],axis='columns', inplace=True)
    return df


if __name__ == "__main__":
    dataset = load_data('bengaluru_house_data.csv')
    dataset = feature_engg(dataset)
    print(dataset.head())
    
    
    