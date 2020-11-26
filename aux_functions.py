################ CUSTOM FUNCTIONS ###########################333
##--------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import itertools 
from pandas.tseries.offsets import MonthEnd

NUM_COLS = ['A_true', 'A_false', 'internal_B_true', 'external_B_true', 'B_true', 'internal_C_true',
            'external_C_true', 'C_true','one_week_B_true_C_true', 'two_weeks_B_true_C_true',
            'one_month_B_true_C_true', 'avg_time_start_C_true', 'median_time_start_C_true',
            'min_time_start_C_true', 'max_time_start_C_true', 'avg_time_B_true_C_true',
            'median_time_B_true_C_true', 'automatization_A', 'automatization_B', 'automatization_C']

RELEVANT_COLS = np.append(['customer', 'contractid','contract_type', 'date', 'startdate', 'enddate',
                           'is_active'], NUM_COLS)

# To create basic exploratory plots
def eda_plots(df, col, title=None, bins=10, density=True):
    '''
    Creates an histogram, scatterplot and boxplot for the data
    ----------------------------------------------------------
    Input:
        df: pd.DataFrame containing data to be plotted.
        col: str, which column from pd.DataFrame
        title. The main title of the figure (str)
        bins: int (default value is 10).
        density: bool, whether normalize the histogram (default values is True)
    Output:
        figure showing the three plots
    '''
    np.warnings.filterwarnings('ignore')
    if df[col].isna().sum() != 0:
        print(df[col].isna().sum(), 'Nan values were not plotted...\n')
        df[col] = df[col].dropna()
    df = df.sort_values(by='date').reset_index(drop=True)
    colors = np.where(df['is_active']=='inactive', 'r', 'b')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle(title)
    ax1.hist(df[col], bins=bins, density=density)
    ax1.set_title('Histogram')
    ax2.scatter(df.index, df[col].values, c=colors, alpha=.4)
    ax2.set_title('Scatter plot')
    ax3.boxplot(df[col].dropna())
    ax3.set_title('Boxplot')
    print(df[col].describe())
    plt.show()

def generate_features_distribution(df):
    '''
    Generates descriptive statistics and eda plots for each feature, and computes the its correlation
    coefficient with the target.
    Input:
        df: pd.DataFrame
    Output:
        prints r coefficient for both cases, droppping and filling missing values, displays descriptive
        statistics table and basic eda plots.
    '''
    for col in df.iloc[:,7:].columns:
        print(col + ":")
        for nan_met in ['fill', 'drop']:
            print(nan_met)
            test_corr(df, col, nan_met)
        eda_plots(df, col, bins=50)
        print('-'*50)

##################################################
# To compute correlation between target and features
def test_corr(df, col='enddate', nan_method='fill'):
    '''
    Computes te correlation coefficient between 'is_active' and another column variable with missing values
    from a DataFrame.
    ---------------------------------------
    Input:
        df: pd.DataFrame
        col: column from df to use
        nan_method: str, nan strategy, it must be either 'fill' (to fill with -1) or 'drop' (to drop)
    Output:
        correlation pd.DataFrame
    '''
    test = pd.DataFrame()
    # datetime
    start_day = np.datetime64('1969-12-31')
    if nan_method not in ['fill', 'drop']:
        print("'nan_method' must be either 'fill' or 'drop'")
        pass
    elif nan_method == 'drop':
        test[col] = df.loc[:, col].dropna()
        if np.dtype(df[col]) == np.dtype('datetime64[ns]'):
            test[col] = test[col].astype('int64') / 10**6
    else:
        if np.dtype(df[col]) == np.dtype('datetime64[ns]'):
            test[col] = df.loc[:,col].fillna(start_day)
            test[col] = test[col].astype('int64') / 10**6
        else:
            test[col] = df.loc[:,col].fillna(df.median())
    test['active'] = (df.loc[:,'is_active'] == 'active').astype('int64')

    print('{:.2f} (non NaNs:{})'.format(test.corr().values[0,1], test.shape[0]))

# To prepare and clean dataset
def split_contract_by_type(df):
    '''
    Splits the 'contractid' into contract_types for 'contractid' values with more than a single type of
    contract
    --------------------------------
    Input:
        df. pd.DataFrame to split
    Output:
        df. pd.DataFrame splitted
    '''
    df_copy = df.copy()
    df_copy.astype({'contractid': 'str'})
    df_types = df_copy[['contractid', 'contract_type']].groupby('contractid').nunique()['contract_type']
    idx = df_types[df_types>1].index
    for i in idx:
        df_copy.loc[(df_copy.contractid == i) & (df_copy.contract_type == 'Brand'),'contractid'] = str(i) + '_brand'
        df_copy.loc[(df_copy.contractid == i) & (df_copy.contract_type == 'Piracy'),'contractid'] = str(i) + '_piracy'
    return df_copy

def year_month_to_int(df, int_type=False):
    '''
    Function to combine 'year' and 'month' columns to create a 'date' column of 'int' type.
    ----------------------------------
    Inputs:
        df: pd.DataFrame, must contain the columns 'year' and 'month'
        int_type: bool, whether set date type to int (default = False)
    Outputs:
        df: pd.DataFrame, the initial dataframe with an additional 'date' column of 'int' type combining
        'year' and 'month'.
    '''
    df_copy = df.copy()
    df_copy.loc[:,'startdate'] = pd.to_datetime(df['startdate'], unit='ms')
    df_copy.loc[:,'enddate'] = pd.to_datetime(df['enddate'], unit='ms')
    df_copy['day'] = np.ones(df_copy.shape[0])
    df_copy['date'] = pd.to_datetime(df_copy[['year','month','day']]) + MonthEnd(1)
    if int_type:
        df['date'] = df['date'].astype('int64') / 10**6 ## to set unit 'ms'
    return df_copy.sort_values(by=['customer', 'contractid', 'date'])

def remove_off_contract(df):
    '''
    Removes records taking place before the 'starttime' and after 'enddate'.
    Input:
        df: pd.DataFrame from which to remove records off the contract period.
    Output:
        clean_df: pd.DataFrame without such records.
    '''
    earlier_mask = (df.date < df.startdate)
    later_mask = (df.date > df.enddate)
    clean_df = df[~(earlier_mask | later_mask)]
    return clean_df


def fix_out_range_values(df):
    '''
    Replaces values from "<time>_B_true_C_true" which are less than 0 or greater than 1.
    Input:
        df: pd.DataFrame
    Output:
        df_fixed: pd.DataFrame with the values replaced by the median.
    '''

    time_B_true_C_true = ['one_week_B_true_C_true', 'two_weeks_B_true_C_true', 'one_month_B_true_C_true']
    conds = [(df[time_B_true_C_true[0]] > 1),
             (df[time_B_true_C_true[1]] > 1),
             (df[time_B_true_C_true[2]] > 1)]
    for feat, cond in zip(time_B_true_C_true, conds):
        df.loc[cond, feat] = df[feat].median()
    df_fixed = df.sort_values(by=['date'])
    return df_fixed


def remove_outliers(df, strong_cond=True):
    '''
    Removes outliers from all numerical variables
    ----------------------------------------------
    Input:
        df: pd.DataFrame.
        strong_cond: bool, whether to remove outliers defined according to strong (Q1-3*IQR, Q3+3*IQR)
        or weak condition (Q1-1.5*IQR, Q3+1.5*IQR).
    Output:
        df_out: pd.DataFrame with rows corresponding to outliers removed.
    '''
    df_out = df.copy()
    if strong_cond:
        gamma = 3
    else:
        gamma = 1.5
    for col in NUM_COLS:
        IQR = df[col].quantile(.75) - df[col].quantile(.25)
        int_plus = (df[col] > df[col].quantile(.75) + gamma * IQR)
        int_minus = (df[col] < df[col].quantile(.25) - gamma * IQR)
        df_out.drop(df_out.loc[(int_plus | int_minus)].index, inplace=True)
    return  df_out


def clean_features(df, cols=RELEVANT_COLS, drop_outdated=True, num_target=True):
    '''
    Kind of a wrap-up function to clean features:
        - Selects the relevant subset of columns,
        - combines 'year' and 'month' into the single new variable 'date',
        - splits the 'contractid' entries with more than a single 'contract_type' value into separated
        'contractid's.
        - replaces 'contract_type' column with the corresponding one-hot encoding.
        - removes records corresponding to 'date' taking place before 'startdate' or after 'enddate'.
    --------------------------------------
    Input:
        df: pd.DataFrame
        cols: list of the column names to be selected. By default 'relevant_cols' below are selected.
        drop_outdated: bool, whether to drop records corresponding to dates before 'startdate' or after
        'enddate'.
        num_target: bool, whether to convert the target to integer [0,1].
    Returns:
        clean_df: pd.DataFrame
    '''
    ## combine year and month into date
    df_with_date = year_month_to_int(df)

    ## drop fields which are not in the Test description, presumably they are not supposed to be used?
    selected_df = df_with_date[cols]

    ## split contracts with not a single contract_type into separated contracts
    split_df = split_contract_by_type(selected_df)

    ## one-hot encoding of contract_type
    split_df['brand'] = (split_df['contract_type'] == 'Brand').astype('int64')
    split_df['piracy'] = (split_df['contract_type'] == 'Piracy').astype('int64')
    split_df.drop(columns='contract_type', inplace=True)
    if num_target:
        split_df['active'] = (split_df.loc[:,'is_active'] == 'active').astype('int64')
        split_df.drop(columns='is_active', inplace=True)
    ## reorder to bring the new one-hot columns into the begining
    idx_list = list(range(len(split_df.columns)))
    order = idx_list[:3] + idx_list[-3:] + idx_list[3:-3]
    cols_order = [split_df.columns[o] for o in order]


    ## correct values out of range
    fix_df = fix_out_range_values(split_df[cols_order])

    ## remove records off contract
    if drop_outdated:
        clean_df = remove_off_contract(fix_df)
    else:
        clean_df = fix_df

    return clean_df

#### Group data per customer/contract

def group_by_contract(df):
    '''
    Groups data per contract
    --------------------------
    Input:
        df: pd.DataFrame to be grouped
    Output:
        clean_by_contract: pd.DataFrame with values grouped by 'contractid', returning the cumulative sum
        of each numerical variable corresponding to the quantiles 0, .25, .5, .75 and 1 of the historic
        record for each variable.
    '''
    def cum_q0(x):
        x_sort = x.sort_values()
        x_cum = x_sort.cumsum()
        return x.min()
    def cum_q1(x):
        x_sort = x.sort_values()
        x_cum = x_sort.cumsum()
        return  x_cum.quantile(.25)
    def cum_q2(x):
        x_sort = x.sort_values()
        x_cum = x_sort.cumsum()
        return x_cum.quantile(.5)
    def cum_q3(x):
        x_sort = x.sort_values()
        x_cum = x_sort.cumsum()
        return x_cum.quantile(.75)
    def cum_q4(x):
        x_sort = x.sort_values()
        x_cum = x_sort.cumsum()
        return x_cum.sum()
    agg_fun = {x: np.unique for x in ['is_active', 'brand', 'piracy']}
    agg_fun.update({col: [cum_q0, cum_q1, cum_q2, cum_q3, cum_q4] for col in NUM_COLS})
    clean_by_contract = df.groupby('contractid').agg(agg_fun)
    return clean_by_contract

## Probably the most inefficient way to generate this dataframe but with pivot some stuff was missing...

def get_data_per_customer(df):
    '''
    Function to build a dataframe with data gathered by contract.
    '''
    # split date dependent and independent columns
    date_ind_cols = df.columns[:8].drop('date')
    date_dep_cols = df.columns[8:]

    # identify the unique contracts
    contract_ids = df['contractid'].unique()
    contracts_cols = np.append(date_ind_cols, [(col + '_' + str(d)[2:7]) for d in df.date.unique()
                                               for col in date_dep_cols])
    contracts_df = pd.DataFrame(index=contract_ids, columns = contracts_cols)

    # check whether the date_independent cols really are so
    for col in date_ind_cols:
        for c in contract_ids:
            if len(df[col][df.contractid == c].unique()) != 1:
                print('{} for contractid {} is not date independent'.format(col, c))
        contracts_df[col] = [df[col][df.contractid == c].unique()[0] for c in df['contractid'].unique()]
    for c in contract_ids:
        for col in date_dep_cols:
            for d in df.date[df.contractid == c]:
                contracts_df.loc[c, col + '_' + str(d)[2:7]] = df[col][(df.contractid == c) & (df.date == d)].values
    return contracts_df
