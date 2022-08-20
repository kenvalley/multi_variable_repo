from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather data
boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)
features.head()

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])
target.shape

property_stats = np.ndarray(shape=(1,11))
property_stats[0][0] = 83
property_stats

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

# challenge: calculate the MSE and RMSE using sklearn
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

CRIME_IDX = 0
CHAX_IDX = 2
ZN_IDX = 1
RM_IDX = 4
PTRATIO_IDX = 8

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE/np.median(boston_dataset.target)


def get_log_estimate(nr_rooms,
                    students_per_classrooms,
                    next_to_river=False,
                    high_confidence=True):
    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classrooms
    
    if next_to_river:
        property_stats[0][CHAX_IDX] = 1
    else:
        property_stats[0][CHAX_IDX] = 0
        
    # Make Prediction
    log_estimate = regr.predict(property_stats)[0][0]
    
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    
    return log_estimate, upper_bound, lower_bound, interval


# Challenge: Write the Pytho code that converts the log price estimate using 
# 1970s Proces as well as the upper and lower bounds 
# to today's prices, Round the values to the nearest 1000 dollars 

log_est, upper, lower, conf = get_log_estimate(9, students_per_classrooms=15, 
                                              next_to_river=False, high_confidence=False)

# Convert to today's dollars
dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
dollar_high = np.e**upper * 1000 * SCALE_FACTOR
dollar_lower = np.e**lower * 1000 * SCALE_FACTOR

# Round the dollar values to nearest thousand
rounded_est = np.around(dollar_est, -3)
rounded_high = np.around(dollar_high, -3)
rounded_lower = np.around(dollar_lower, -3)

print(f'The estimated property value is {rounded_est}.')
print(f'At {conf}% confidence, the valuation range is')
print(f'USD {rounded_lower} at the lower end to USD {rounded_high} at the high end')

def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    
    """
    Estimate the price of a property in Boston.
    
    Keyword arguments:
    rm -- number of rooms in the property
    ptratio -- number of students per teacher in the classroo for the school in the area
    chas -- True if the property is next to the river, False otherwise
    large_range -- True for a 95% prediction interval, False for a 68% interval
    
    """
    
    
    if rm<1 or ptratio<1:
        print('That is unrealistic. Try again!')
        return
    
    
    log_est, upper, lower, conf = get_log_estimate(rm, 
                                                   ptratio, 
                                                   next_to_river=chas, high_confidence=large_range)

    # Convert to today's dollars
    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_high = np.e**upper * 1000 * SCALE_FACTOR
    dollar_lower = np.e**lower * 1000 * SCALE_FACTOR

    # Round the dollar values to nearest thousand
    rounded_est = np.around(dollar_est, -3)
    rounded_high = np.around(dollar_high, -3)
    rounded_lower = np.around(dollar_lower, -3)

    print(f'The estimated property value is {rounded_est}.')
    print(f'At {conf}% confidence, the valuation range is')
    print(f'USD {rounded_lower} at the lower end to USD {rounded_high} at the high end')   
     