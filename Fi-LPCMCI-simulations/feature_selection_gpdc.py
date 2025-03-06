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


def compute_f1_score(ground_truth, predicted):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    equivalences = {
    '-->': '-?>',
    '<--': '<?-',
    'o-o': {'<-o', 'o->', '<->', '<?>', 'o?o'}
}
    
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[1]):
            ground_relation = ground_truth[i][j]
            predicted_relation = predicted[i][j]
            
            for t in range(2): 
                g_rel = ground_relation[t]
                p_rel = predicted_relation[t]
                
                if isinstance(g_rel, np.ndarray):
                    g_rel = str(g_rel)
                if isinstance(p_rel, np.ndarray):
                    p_rel = str(p_rel)
                
                if g_rel == '' and p_rel == '':
                    continue
                elif g_rel == p_rel:
                    true_positives += 1
                elif g_rel in equivalences:
                    if p_rel in equivalences[g_rel] if isinstance(equivalences[g_rel], set) else p_rel == equivalences[g_rel]:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    if p_rel != '':
                        false_positives += 1
                    if g_rel != '':
                        false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1_score




def extract_cmi(graphical_array, values_array):
    cmi_values = []
    
    valid_arrows = {'-->', '<--', 'o-o', '-?>', '<?-', '<->', 'o->', '<-o'}
    
    for i in range(graphical_array.shape[0]):
        for j in range(graphical_array.shape[1]):
            ground_relation = graphical_array[i][j]
            value_pair = values_array[i][j]
            
            for relation, value in zip(ground_relation, value_pair):
                if relation in valid_arrows:
                    if np.isfinite(value):
                        cmi_values.append(value)
    
    if len(cmi_values) > 0:
        avg_cmi = np.mean(cmi_values)
    else:
        avg_cmi = 0 
    
    return avg_cmi


def latent_link_recall(ground_truth, predicted):

    latent_links = {'o-o', '<-o', 'o->', '<->'}
    
    true_positives = 0
    false_negatives = 0
    
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[1]):
            ground_relation = ground_truth[i][j]
            predicted_relation = predicted[i][j]
            
            for g_rel, p_rel in zip(ground_relation, predicted_relation):
                if g_rel in latent_links:
                    if p_rel in latent_links:
                        true_positives += 1
                    else:
                        false_negatives += 1
    
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    
    return recall

def compute_f1_score(ground_truth, predicted):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    equivalences = {
    '-->': '-?>',
    '<--': '<?-',
    'o-o': {'<-o', 'o->', '<->', '<?>', 'o?o'}
}
    
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[1]):
            ground_relation = ground_truth[i][j]
            predicted_relation = predicted[i][j]
            
            for t in range(2):
                g_rel = ground_relation[t]
                p_rel = predicted_relation[t]
                
                if isinstance(g_rel, np.ndarray):
                    g_rel = str(g_rel)
                if isinstance(p_rel, np.ndarray):
                    p_rel = str(p_rel)
                
                if g_rel == '' and p_rel == '':
                    continue
                elif g_rel == p_rel:
                    true_positives += 1
                elif g_rel in equivalences:
                    if p_rel in equivalences[g_rel] if isinstance(equivalences[g_rel], set) else p_rel == equivalences[g_rel]:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    if p_rel != '':
                        false_positives += 1
                    if g_rel != '':
                        false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1_score




def extract_cmi(graphical_array, values_array):
    cmi_values = []
    
    valid_arrows = {'-->', '<--', 'o-o', '-?>', '<?-', '<->', 'o->', '<-o'}
    
    for i in range(graphical_array.shape[0]):
        for j in range(graphical_array.shape[1]):
            ground_relation = graphical_array[i][j]
            value_pair = values_array[i][j]
            
            for relation, value in zip(ground_relation, value_pair):
                if relation in valid_arrows:
                    if np.isfinite(value):
                        cmi_values.append(value)
    
    if len(cmi_values) > 0:
        avg_cmi = np.mean(cmi_values)
    else:
        avg_cmi = 0
    
    return avg_cmi


def latent_link_recall(ground_truth, predicted):
    latent_links = {'o-o', '<-o', 'o->', '<->'}
    
    true_positives = 0
    false_negatives = 0
    
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[1]):
            ground_relation = ground_truth[i][j]
            predicted_relation = predicted[i][j]
            
            for g_rel, p_rel in zip(ground_relation, predicted_relation):
                if g_rel in latent_links:
                    if p_rel in latent_links:
                        true_positives += 1
                    else:
                        false_negatives += 1
    
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    
    return recall

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
        #cmi_knn = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks', sig_samples=200)
        gpdc = GPDC(significance='analytic', gp_params=None)
        print("Starting LPCMCI...")
        lpcmci_loc = LPCMCI(
            dataframe=data, 
            cond_ind_test=gpdc,
            verbosity=0)
        results = lpcmci_loc.run_lpcmci(tau_max=1, pc_alpha=.2, link_assumptions = link_assumptions, n_preliminary_iterations = 0)
        end = time.time()
        #tp.plot_time_series_graph(graph=results['graph'],
        #                        val_matrix=results['val_matrix'])
        if len(gt) == 0:
            f1 = 0
        else:
            f1 = compute_f1_score(gt, results['graph'])

        cmi_val = extract_cmi(results['graph'], results['val_matrix'])
        recall_latent = latent_link_recall(gt, results['graph'])
        elapsed_time = end - start
        print("Finished.")
        return results, elapsed_time, f1, cmi_val, recall_latent

elapsed_time = []
f1 = []
cmi = []
recall = []

# Function to generate a dataset and its ground truth causal matrix
def generate_dataset(n, nan_ratio=0.1, phi1=0.8, phi2=0.7):

    # Initialize latent confounders
    latent1 = np.zeros(n)
    latent2 = np.zeros(n)
    latent1[0] = np.random.normal()
    latent2[0] = np.random.normal()

    # Generate latent confounders using AR(1) processes
    for t in range(1, n):
        latent1[t] = phi1 * latent1[t-1] + np.random.normal()
        latent2[t] = phi2 * latent2[t-1] + np.random.normal()

    # Initialize system variables y1 to y5
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)
    y4 = np.zeros(n)
    y5 = np.zeros(n)

    # Generate y1 influenced by y2 (one time lag) and latent1
    y1[0] = np.random.normal()
    for t in range(1, n):
        y1[t] = 0.5 * y2[t-1] + 0.6 * latent1[t] + np.random.normal()

    # Generate y2 using AR(1) process (y2 causes itself over time)
    y2[0] = np.random.normal()
    for t in range(1, n):
        y2[t] = 0.7 * y2[t-1] + 0.8 * latent2[t] + np.random.normal()

    # Generate y3 influenced by latent2
    y3[0] = np.random.normal()
    for t in range(1, n):
        y3[t] = 0.8 * latent2[t] + np.random.normal()

    # Generate y4 influenced by y3 (one time lag) and latent1
    y4[0] = np.random.normal()
    for t in range(1, n):
        y4[t] = 0.8 * y3[t-1] + 0.5 * latent1[t] + np.random.normal()

    # Generate y5 influenced by y4 (contemporaneous), self-lag, and latent2
    y5[0] = np.random.normal()
    for t in range(1, n):
        y5[t] = 0.6 * y5[t-1] + 0.9 * y4[t] + np.random.normal()

    # Introduce NaN values randomly
    nan_indices = np.random.choice(n, size=int(n * nan_ratio), replace=False)
    for y in [y1, y2, y3, y4, y5]:
        y[nan_indices] = np.nan

    # Combine into DataFrame
    data = pd.DataFrame({'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'y5': y5})

    # Ground truth causal matrix
    ground_truth = np.array([
        # y1 caused by y2[t-1] and latent1
        [['', ''],  # y1 not influencing anything
         ['', ''],  # y1 caused by y2[t-1]
         ['', ''],     # No relation with y3
         ['o-o', ''],  # y1 caused by latent1
         ['', '']],    # No relation with y5

        # y2 causes itself over time (AR(1)) and causes y1[t-1]
        [['', '-->'],    # y2 not caused by y1
         ['', '-->'],  # y2 causes itself at lag 1
         ['o-o', ''],  # y2 caused by latent2
         ['', ''],    # No relation with y4
         ['', '']],   # No relation with y5

        # y3 caused by latent2, causes y4[t-1]
        [['', ''],    # No relation with y1
         ['o-o', ''], # y3 caused by latent2
         ['', ''],    # y3 not influenced by itself
         ['', '-->'], # y3[t-1] causes y4
         ['', '']],   # No relation with y5

        # y4 caused by y3[t-1] and latent1, causes y5 contemporaneously
        [['o-o', ''],    # No relation with y1
         ['', ''],    # No relation with y2
         ['', ''], # y4 caused by y3[t-1]
         ['-->', ''],    # y4 not influenced by itself
         ['-->', '']], # y4 causes y5 contemporaneously

        # y5 caused by y4 contemporaneously and self at lag
        [['', ''],    # No relation with y1
         ['', ''],    # No relation with y2
         ['o-o', ''], # y5 caused by latent2
         ['<--', ''], # y5 caused by y4 contemporaneously
         ['', '-->']]  # y5 causes itself at lag
    ])

    return data, ground_truth

# Example usage:
n = [5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]  # Length of the time series
nan_ratio = 0.1  # Ratio of NaN values
num_datasets = 3  # Number of datasets

elapsed_time = []
f1 = []
cmi = []
recall = []
for j in n:
    for i in range(num_datasets):
        data, ground_truth = generate_dataset(j, nan_ratio)
        result = feature_selection_lpcmci(data, ground_truth)
        elapsed_time.append(result[1])
        f1.append(result[2])
        cmi.append(result[3])
        recall.append(result[4])


# Create DataFrame
df_1 = pd.DataFrame({
    "Data": [item for item in n for _ in range(num_datasets)],
    'Elapsed Time (s)': elapsed_time,
    'F1 Score': f1,
    'CMI': cmi,
    'Recall': recall
})

# Print the DataFrame
print(df_1)
df_1.to_csv('/home/gnicolaou/tigramite/tigramite/simulations_results/feature_selection_gpdc_5_extended.csv')

# Function to generate the dataset and its ground truth causal matrix for a 10-node system
def generate_causal_dataset(n, nan_ratio=0.1, phi_latent=0.8):
    # Initialize latent confounder
    latent1 = np.zeros(n)
    latent1[0] = np.random.normal()

    # Generate latent confounder using an AR(1) process
    for t in range(1, n):
        latent1[t] = phi_latent * latent1[t - 1] + np.random.normal()

    # Initialize variables y1 to y10
    y = {f'y{i}': np.zeros(n) for i in range(1, 11)}

    # Define the causal relationships
    # y2 influenced by latent1
    y['y2'][0] = np.random.normal()
    for t in range(1, n):
        y['y2'][t] = 0.6 * latent1[t] + np.random.normal()

    # y1 influenced by y2 (contemporaneous)
    y['y1'][0] = np.random.normal()
    for t in range(1, n):
        y['y1'][t] = 0.5 * y['y2'][t] + np.random.normal()

    # y3 influenced by y1[t-1]
    y['y3'][0] = np.random.normal()
    for t in range(1, n):
        y['y3'][t] = 0.7 * y['y1'][t - 1] + np.random.normal()

    # y4 influenced by y2[t-1] and y3[t] (contemporaneous)
    y['y4'][0] = np.random.normal()
    for t in range(1, n):
        y['y4'][t] = 0.5 * y['y2'][t - 1] + 0.8 * y['y3'][t] + np.random.normal()

    # y5 influenced by latent1 and self (y5[t-1])
    y['y5'][0] = np.random.normal()
    for t in range(1, n):
        y['y5'][t] = 0.6 * latent1[t] + 0.5 * y['y5'][t - 1] + np.random.normal()

    # Additional nodes y6 to y10 with new relationships
    # y6 influenced by y5[t-1] and latent1
    y['y6'][0] = np.random.normal()
    for t in range(1, n):
        y['y6'][t] = 0.7 * y['y5'][t - 1] + 0.4 * latent1[t] + np.random.normal()

    # y7 influenced by y6[t] and y2[t-1]
    y['y7'][0] = np.random.normal()
    for t in range(1, n):
        y['y7'][t] = 0.5 * y['y6'][t] + 0.3 * y['y2'][t - 1] + np.random.normal()

    # y8 influenced by y3[t-1] and y7[t]
    y['y8'][0] = np.random.normal()
    for t in range(1, n):
        y['y8'][t] = 0.6 * y['y3'][t - 1] + 0.5 * y['y7'][t] + np.random.normal()

    # y9 influenced by y4[t] and latent1
    y['y9'][0] = np.random.normal()
    for t in range(1, n):
        y['y9'][t] = 0.4 * y['y4'][t] + 0.6 * latent1[t] + np.random.normal()

    # y10 influenced by y8[t] and y9[t-1]
    y['y10'][0] = np.random.normal()
    for t in range(1, n):
        y['y10'][t] = 0.5 * y['y8'][t] + 0.7 * y['y9'][t - 1] + np.random.normal()

    # Introduce NaN values randomly
    nan_indices = np.random.choice(n, size=int(n * nan_ratio), replace=False)
    for i in range(1, 11):
        y[f'y{i}'][nan_indices] = np.nan

    # Combine into DataFrame
    data = pd.DataFrame(y)

    # Define ground truth causal matrix (updated for 10 nodes)
    ground_truth = np.array([
        [['', ''], ['<--', ''], ['', '-->'], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', '']],
        [['-->', ''], ['', ''], ['', ''], ['', '-->'], ['o-o', ''], ['o-o', ''], ['', '-->'], ['', ''], ['o-o', ''], ['', '']],
        [['', '<--'], ['', ''], ['', ''], ['-->', ''], ['', ''], ['', ''], ['', ''], ['', '-->'], ['', ''], ['', '']],
        [['', ''], ['', '<--'], ['<--', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['-->', ''], ['', '']],
        [['', ''], ['o-o', ''], ['', ''], ['', ''], ['', '-->'], ['', '-->'], ['', ''], ['', ''], ['o-o', ''], ['', '']],
        [['', ''], ['o-o', ''], ['', ''], ['', ''], ['', '<--'], ['', ''], ['-->', ''], ['', ''], ['o-o', ''], ['', '']],
        [['', ''], ['', '<--'], ['', ''], ['', ''], ['', ''], ['<--', ''], ['', ''], ['-->', ''], ['', ''], ['', '']],
        [['', ''], ['', ''], ['', '<--'], ['', ''], ['', ''], ['', ''], ['<--', ''], ['', ''], ['', ''], ['-->', '']],
        [['', ''], ['o-o', ''], ['', ''], ['<--', ''], ['o-o', ''], ['o-o', ''], ['', ''], ['', ''], ['', ''], ['', '-->']],
        [['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['<--', ''], ['', '<--'], ['', '']]
    ])

    return data, ground_truth

elapsed_time = []
f1 = []
cmi = []
recall = []
for j in n:
    for i in range(num_datasets):
        data, ground_truth = generate_dataset(j, nan_ratio)
        result = feature_selection_lpcmci(data, ground_truth)
        elapsed_time.append(result[1])
        f1.append(result[2])
        cmi.append(result[3])
        recall.append(result[4])


# Create DataFrame
df_2 = pd.DataFrame({
    "Data": [item for item in n for _ in range(num_datasets)],
    'Elapsed Time (s)': elapsed_time,
    'F1 Score': f1,
    'CMI': cmi,
    'Recall': recall
})

# Print the DataFrame
print(df_2)
df_2.to_csv('/home/gnicolaou/tigramite/tigramite/simulations_results/feature_selection_gpdc_10_extended.csv')

import numpy as np
import pandas as pd

# Function to generate the dataset and its ground truth causal matrix for a 15-node system
def generate_causal_dataset(n, nan_ratio=0.1, phi_latent=0.8):
    # Initialize latent confounder
    latent1 = np.zeros(n)
    latent1[0] = np.random.normal()

    # Generate latent confounder using an AR(1) process
    for t in range(1, n):
        latent1[t] = phi_latent * latent1[t - 1] + np.random.normal()

    # Initialize variables y1 to y15
    y = {f'y{i}': np.zeros(n) for i in range(1, 16)}

    # Define the causal relationships
    # y2 influenced by latent1
    y['y2'][0] = np.random.normal()
    for t in range(1, n):
        y['y2'][t] = 0.6 * latent1[t] + np.random.normal()

    # y1 influenced by y2 (contemporaneous)
    y['y1'][0] = np.random.normal()
    for t in range(1, n):
        y['y1'][t] = 0.5 * y['y2'][t] + np.random.normal()

    # y3 influenced by y1[t-1]
    y['y3'][0] = np.random.normal()
    for t in range(1, n):
        y['y3'][t] = 0.7 * y['y1'][t - 1] + np.random.normal()

    # y4 influenced by y2[t-1] and y3[t] (contemporaneous)
    y['y4'][0] = np.random.normal()
    for t in range(1, n):
        y['y4'][t] = 0.5 * y['y2'][t - 1] + 0.8 * y['y3'][t] + np.random.normal()

    # y5 influenced by latent1 and self (y5[t-1])
    y['y5'][0] = np.random.normal()
    for t in range(1, n):
        y['y5'][t] = 0.6 * latent1[t] + 0.5 * y['y5'][t - 1] + np.random.normal()

    # Additional nodes y6 to y15 with new relationships
    # y6 influenced by y5[t-1] and latent1
    y['y6'][0] = np.random.normal()
    for t in range(1, n):
        y['y6'][t] = 0.7 * y['y5'][t - 1] + 0.4 * latent1[t] + np.random.normal()

    # y7 influenced by y6[t] and y2[t-1]
    y['y7'][0] = np.random.normal()
    for t in range(1, n):
        y['y7'][t] = 0.5 * y['y6'][t] + 0.3 * y['y2'][t - 1] + np.random.normal()

    # y8 influenced by y3[t-1] and y7[t]
    y['y8'][0] = np.random.normal()
    for t in range(1, n):
        y['y8'][t] = 0.6 * y['y3'][t - 1] + 0.5 * y['y7'][t] + np.random.normal()

    # y9 influenced by y4[t] and latent1
    y['y9'][0] = np.random.normal()
    for t in range(1, n):
        y['y9'][t] = 0.4 * y['y4'][t] + 0.6 * latent1[t] + np.random.normal()

    # y10 influenced by y8[t] and y9[t-1]
    y['y10'][0] = np.random.normal()
    for t in range(1, n):
        y['y10'][t] = 0.5 * y['y8'][t] + 0.7 * y['y9'][t - 1] + np.random.normal()

    # y11 influenced by y9 and latent1
    y['y11'][0] = np.random.normal()
    for t in range(1, n):
        y['y11'][t] = 0.3 * y['y9'][t] + 0.5 * latent1[t] + np.random.normal()

    # y12 influenced by y10[t-1] and y5[t]
    y['y12'][0] = np.random.normal()
    for t in range(1, n):
        y['y12'][t] = 0.4 * y['y10'][t - 1] + 0.6 * y['y5'][t] + np.random.normal()

    # y13 influenced by y12 and y7[t-1]
    y['y13'][0] = np.random.normal()
    for t in range(1, n):
        y['y13'][t] = 0.5 * y['y12'][t] + 0.3 * y['y7'][t - 1] + np.random.normal()

    # y14 influenced by y8[t] and y13[t-1]
    y['y14'][0] = np.random.normal()
    for t in range(1, n):
        y['y14'][t] = 0.6 * y['y8'][t] + 0.4 * y['y13'][t - 1] + np.random.normal()

    # y15 influenced by latent1 and y14[t]
    y['y15'][0] = np.random.normal()
    for t in range(1, n):
        y['y15'][t] = 0.7 * latent1[t] + 0.3 * y['y14'][t] + np.random.normal()

    # Introduce NaN values randomly
    nan_indices = np.random.choice(n, size=int(n * nan_ratio), replace=False)
    for i in range(1, 16):
        y[f'y{i}'][nan_indices] = np.nan

    # Combine into DataFrame
    data = pd.DataFrame(y)

    # Define ground truth causal matrix (updated for 15 nodes)
    ground_truth = np.array([
        [['', ''], ['<--', ''], ['', '-->'], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', '']],
        [['-->', ''], ['', ''], ['', ''], ['', '-->'], ['o-o', ''], ['o-o', ''], ['', '-->'], ['', ''], ['o-o', ''], ['', ''], ['o-o', ''], ['', ''], ['', ''], ['', ''], ['o-o', '']],
        [['', '<--'], ['', ''], ['', ''], ['-->', ''], ['', ''], ['', ''], ['', ''], ['', '-->'], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', '']],
        [['', ''], ['', '<--'], ['<--', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['-->', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', '']],
        [['', ''], ['o-o', ''], ['', ''], ['', ''], ['', '-->'], ['', '-->'], ['', ''], ['', ''], ['o-o', ''], ['', ''], ['o-o', ''], ['-->', ''], ['', ''], ['', ''], ['o-o', '']],
        [['', ''], ['o-o', ''], ['', ''], ['', ''], ['', '<--'], ['', ''], ['-->', ''], ['', ''], ['o-o', ''], ['', ''], ['o-o', ''], ['', ''], ['', ''], ['', ''], ['o-o', '']],
        [['', ''], ['', '<--'], ['', ''], ['', ''], ['', ''], ['<--', ''], ['', ''], ['-->', ''], ['', ''], ['', ''], ['', ''], ['', '-->'], ['', ''], ['', ''], ['', '']],
        [['', ''], ['', ''], ['', '<--'], ['', ''], ['', ''], ['', ''], ['<--', ''], ['', ''], ['', ''], ['-->', ''], ['', ''], ['', ''], ['', ''], ['-->', ''], ['', '']],
        [['', ''], ['o-o', ''], ['', ''], ['<--', ''], ['o-o', ''], ['o-o', ''], ['', ''], ['', ''], ['', ''], ['', '-->'], ['-->', ''], ['', ''], ['', ''], ['', ''], ['o-o', '']],
        [['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['<--', ''], ['', '<--'], ['', ''], ['', ''], ['', '-->'], ['', ''], ['', ''], ['', '']],
        [['', ''], ['o-o', ''], ['', ''], ['', ''], ['o-o', ''], ['o-o', ''], ['', ''], ['', ''], ['<--', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['o-o', '']],
        [['', ''], ['', ''], ['', ''], ['', ''], ['<--', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', '<--'], ['', ''], ['', ''], ['-->', ''], ['', ''], ['', '']],
        [['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', '<--'], ['', ''], ['', ''], ['', ''], ['', ''], ['<--', ''], ['', ''], ['', '-->'], ['', '']],
        [['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['<--', ''], ['', ''], ['', ''], ['', ''], ['', ''], ['', '<--'], ['', ''], ['-->', '']],
        [['', ''], ['o-o', ''], ['', ''], ['', ''], ['o-o', ''], ['o-o', ''], ['', ''], ['', ''], ['o-o', ''], ['', ''], ['o-o', ''], ['', ''], ['', ''], ['<--', ''], ['', '']]
    ])

    return data, ground_truth

elapsed_time = []
f1 = []
cmi = []
recall = []
for j in n:
    for i in range(num_datasets):
        data, ground_truth = generate_dataset(j, nan_ratio)
        result = feature_selection_lpcmci(data, ground_truth)
        elapsed_time.append(result[1])
        f1.append(result[2])
        cmi.append(result[3])
        recall.append(result[4])


# Create DataFrame
df_3 = pd.DataFrame({
    "Data": [item for item in n for _ in range(num_datasets)],
    'Elapsed Time (s)': elapsed_time,
    'F1 Score': f1,
    'CMI': cmi,
    'Recall': recall
})

# Print the DataFrame
print(df_3)
df_3.to_csv('/home/gnicolaou/tigramite/tigramite/simulations_results/feature_selection_gpdc_15_extended.csv')
