import pandas as pd
import numpy as np

def create_dummy_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Avg. Area Income': np.random.normal(60000, 10000, n_samples),
        'Avg. Area House Age': np.random.normal(6, 2, n_samples),
        'Avg. Area Number of Rooms': np.random.normal(7, 1, n_samples),
        'Avg. Area Number of Bedrooms': np.random.normal(4, 1, n_samples),
        'Area Population': np.random.normal(40000, 10000, n_samples),
        'Address': ['Fake Address ' + str(i) for i in range(n_samples)]
    }
    
    # Linear combination + noise for Price
    price = (
        0.5 * data['Avg. Area Income'] + 
        10000 * data['Avg. Area House Age'] + 
        5000 * data['Avg. Area Number of Rooms'] + 
        2000 * data['Avg. Area Number of Bedrooms'] + 
        0.1 * data['Area Population'] + 
        np.random.normal(0, 5000, n_samples)
    )
    data['Price'] = price
    
    df = pd.DataFrame(data)
    df.to_csv('housing.csv', index=False)
    print("Created housing.csv with", n_samples, "samples.")

if __name__ == "__main__":
    create_dummy_data()
