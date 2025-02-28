import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import numpy as np
from statsmodels.tsa.stattools import adfuller,kpss
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.lpcmci import LPCMCI
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from statsmodels.tools.sm_exceptions import InterpolationWarning
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.lpcmci import LPCMCI
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.model_selection import RandomizedSearchCV
from tigramite.independence_tests.gpdc import GPDC
import torch


def feature_selection_lpcmci(data, gt = [], lag=1, test_size=0.2, random_state=42, alpha=0.05, n_shuffles=100):

        def create_lagged_features(df, column, lags):
            lagged_df = pd.DataFrame()
            lagged_df[column] = df[column]  # Include the original unlagged version
            for lag in range(1, lags + 1):
                lagged_df[f'{column}_lag{lag}'] = df[column].shift(lag)
            lagged_df.dropna()
            return lagged_df


        def shuffle_column(df, column):
            shuffled_df = df.copy()
            shuffled_df = shuffled_df[[column]]
            shuffled_df[column] = np.random.permutation(df[column].values)
            return shuffled_df


        def evaluate_significance(m_real, m_shuffles, alpha):
            mu, sigma = norm.fit(m_shuffles)
            p_value = norm.cdf(m_real, mu, sigma)
            return p_value < alpha


        def calculate_sma(series, window):
            return series.rolling(window=window, min_periods=1).mean()

        num_vars = data.shape[1]
        link_assumptions = {j: {(i, -tau): '' for i in range(num_vars) for tau in range(2) if (i, -tau) != (j, 0)} for j in range(num_vars)}

        start = time.time()
        for target in data.columns:
            potential_drivers = [col for col in data.columns]
            window = round(len(data) * .2)


            sma_baseline = calculate_sma(data[target], window).dropna()
            aligned_data = data.iloc[window-1:].reset_index(drop=True)


            sma_baseline = sma_baseline.reset_index(drop=True)
            y_baseline = aligned_data[target].reset_index(drop=True)


            not_nan_index = y_baseline.dropna().index
            sma_baseline = sma_baseline.loc[not_nan_index]
            y_baseline = y_baseline.dropna()


            X_baseline = sma_baseline.to_frame()


            X_train, X_test, y_train, y_test = train_test_split(X_baseline, y_baseline, test_size=test_size, random_state=random_state)
            baseline_model = RandomForestRegressor(random_state=random_state)
            baseline_model.fit(X_train, y_train)
            baseline_predictions = baseline_model.predict(X_test)
            current_baseline_mse = mean_squared_error(y_test, baseline_predictions)


            definite_drivers = []
            discovered_drivers = []
            definite_non_drivers = []
            aligned_data = data.dropna()
            combined_features_ = pd.DataFrame()


            while potential_drivers:
                current_driver = potential_drivers.pop(0)
                driver_lagged = create_lagged_features(aligned_data, current_driver, lag).dropna()
                best_mse = current_baseline_mse
                best_lag = None
            
                for lag_num in tqdm(range(0, lag + 1), desc=f'Evaluating {current_driver} lags'):
                    if lag_num == 0 and current_driver == target:
                        continue
                    elif lag_num == 0:
                        current_driver_lagged = driver_lagged[[current_driver]]
                    else:
                        current_driver_lagged = driver_lagged[[f'{current_driver}_lag{lag_num}']]
                    '''
                    combined_features = pd.concat([combined_features_, current_driver_lagged], axis=1)
                    target_column = aligned_data[target].loc[combined_features.index]
                    X_train, X_test, y_train, y_test = train_test_split(combined_features, target_column, test_size=test_size, random_state=random_state)
                    combined_model = RandomForestRegressor(random_state=random_state)
                    combined_model.fit(X_train, y_train)
                
                    combined_predictions = combined_model.predict(X_test)
                    combined_mse = mean_squared_error(y_test, combined_predictions)
                    param_dist = {
                        'n_estimators': [50, 100, 200, 300, 400, 500],
                        'max_depth': [None, 10, 20, 30, 40, 50],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False]
                    }
                    '''
                    param_dist = {
                        'n_estimators': [50, 100, 200, 300, 400, 500],
                        'max_depth': [None, 10, 20, 30, 40, 50],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2'],
                        'bootstrap': [True, False]
                    }
                    
                    random_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=random_state),
                                                    param_distributions=param_dist,
                                                    n_iter=100, 
                                                    cv=3,
                                                    verbose=0, 
                                                    random_state=random_state,
                                                    n_jobs=-1)

                    combined_features = pd.concat([combined_features_, current_driver_lagged], axis=1)
                    target_column = aligned_data[target].loc[combined_features.index]
                    X_train, X_test, y_train, y_test = train_test_split(combined_features, target_column, test_size=test_size, random_state=random_state)

                    random_search.fit(X_train, y_train)

                    best_model = random_search.best_estimator_

                    combined_predictions = best_model.predict(X_test)
                    combined_mse = mean_squared_error(y_test, combined_predictions)

                    
                    if combined_mse < best_mse:
                        best_mse = combined_mse
                        best_lag = lag_num
                
                if best_lag is not None:
                    shuffle_mse = []

                    for _ in range(n_shuffles):
                        if best_lag == 0:
                            shuffled_data = shuffle_column(driver_lagged, current_driver)
                        else:
                            shuffled_data = shuffle_column(driver_lagged, f'{current_driver}_lag{best_lag}')

                        shuffled_features = pd.concat([combined_features_, shuffled_data], axis=1)
                        X_train_shuffled, X_test_shuffled, y_train_shuffled, y_test_shuffled = train_test_split(shuffled_features, target_column, test_size=test_size, random_state=random_state)

                        random_search.fit(X_train_shuffled, y_train_shuffled)

                        best_model = random_search.best_estimator_
                        
                        shuffled_predictions = best_model.predict(X_test_shuffled)
                        shuffle_mse.append(mean_squared_error(y_test_shuffled, shuffled_predictions))
                
                    if evaluate_significance(best_mse, shuffle_mse, alpha):
                        if best_lag == 0:
                            definite_drivers.append(current_driver)
                        else:
                            definite_drivers.append(f'{current_driver}_lag{best_lag}')
                        current_baseline_mse = best_mse
                        combined_features_ = pd.concat([combined_features_, current_driver_lagged], axis=1)
                    else:
                        discovered_drivers.append(current_driver if best_lag == 0 else f'{current_driver}_lag{best_lag}')
                else:
                    definite_non_drivers.append(current_driver)

            print(f"Target: {target}")
            print(f"Definite Drivers: {definite_drivers}")
            print(f"Discovered Drivers: {discovered_drivers}")
            print(f"Definite Non-Drivers: {definite_non_drivers}")
            print(f"Final Baseline MSE: {current_baseline_mse}")

            target_idx = data.columns.get_loc(target)
            for driver in definite_drivers:
                if '_lag' in driver:
                    driver_name, lag_value = driver.rsplit('_lag', 1)
                    driver_idx = data.columns.get_loc(driver_name)
                    lag_value = -int(lag_value)
                    link_assumptions[target_idx][(driver_idx, lag_value)] = '-->'
                    #link_assumptions[driver_idx][(target_idx, lag_value)] = '<--'
                else:
                    driver_idx = data.columns.get_loc(driver)
                    link_assumptions[target_idx][(driver_idx, 0)] = '-->'
                    link_assumptions[driver_idx][(target_idx, 0)] = '<--'

            for driver in discovered_drivers:
                if '_lag' in driver:
                    driver_name, lag_value = driver.rsplit('_lag', 1)
                    driver_idx = data.columns.get_loc(driver_name)
                    lag_value = -int(lag_value)
                    link_assumptions[target_idx][(driver_idx, lag_value)] = '-?>'
                    #link_assumptions[driver_idx][(target_idx, lag_value)] = '<?-'
                else:
                    driver_idx = data.columns.get_loc(driver)
                    link_assumptions[target_idx][(driver_idx, 0)] = '-?>'
                    link_assumptions[driver_idx][(target_idx, 0)] = '<?-'

        for target in data.columns:
            target_idx = data.columns.get_loc(target)
            for driver in discovered_drivers + definite_drivers:
                if '_lag' in driver:
                    driver_name, lag_value = driver.rsplit('_lag', 1)
                    lag_value = -int(lag_value)
                    driver_idx = data.columns.get_loc(driver_name)
                    if target_idx != driver_idx: 
                        if link_assumptions[driver_idx][(target_idx, lag_value)] in ['-?>', '-->'] and link_assumptions[target_idx][(driver_idx, 0)] in ['<?-', '<--']:
                            link_assumptions[target_idx][(driver_idx, 0)] = 'o?o'
                            link_assumptions[driver_idx][(target_idx, 0)] = 'o?o'
                else: 
                    driver_idx = data.columns.get_loc(driver)
                    if target_idx != driver_idx: 
                        if link_assumptions[driver_idx][(target_idx, 0)] in ['-?>', '-->'] and link_assumptions[target_idx][(driver_idx, 0)] in ['<?-', '<--']:
                            link_assumptions[target_idx][(driver_idx, 0)] = 'o?o'
                            link_assumptions[driver_idx][(target_idx, 0)] = 'o?o'
        print(link_assumptions)
        n_a_n = np.isnan(data).any(axis=1)
        data[n_a_n] = 999
        num_columns = data.shape[1]
        data = data.values
        data = pp.DataFrame(data, var_names = [f'y{i}' for i in range(num_columns)], missing_flag = 999)
        cmi_knn = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks', sig_samples=200)
        print("Starting LPCMCI...")
        lpcmci_loc = LPCMCI(
            dataframe=data, 
            cond_ind_test=cmi_knn,
            verbosity=0)
        results = lpcmci_loc.run_lpcmci(tau_max=1, pc_alpha=.2, link_assumptions = link_assumptions, n_preliminary_iterations = 0)
        end = time.time()
        elapsed_time = end - start
        print("Finished.")
        return results, elapsed_time

# Load Data, Handle Missingness:
data = torch.load(f"/home/gnicolaou/tigramite/tutorials/causal_discovery/combined_tensor.pt", map_location=torch.device('cpu'))

# Convert to DataFrame
dat = pd.DataFrame(data.numpy())
dat.columns = dat.columns.astype(str)
#print(dat.head())
result = feature_selection_lpcmci(dat)
print(result)




