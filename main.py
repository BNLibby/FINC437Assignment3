
# Imports
import pickle as pk
import os.path as path
import pandas as pd
import statsmodels.api as st
import numpy as np

# TODO: Create 2-stage regression for the following:
#   Market Model
#   3-Factor Model
#   5-Factor Model
#   6-Factor Model
#   *Note: For each model must do 2 separate tests:
#   1.) Use monthly data
#   2.) Use daily data with 1-year moving window
#   *Note: For each test must report t-stat and lambdas

# TODO: Create overall regression, see Assignment3Overview.pdf Part C...good luck


# Configure dictionary of dataframes for data, convert to pickle object
def config_data(config_file, config_path, config_data_type, config_param):
    pk_dict = {}  # Create dictionary to store DataFrames

    # Convert CSVData into DataFrames and store in dictionary
    for name in config_param:
        pk_dict[name] = pd.read_csv(config_path + name + config_data_type, dtype=object)

    # Send dictionary object to pickle file for serialization
    with open(config_file, "wb") as pk_file:
        pk.dump(pk_dict, pk_file)


# Market Model
def market_model(pk_dict, ports_to_test):
    x_axis_data: pd.DataFrame = pk_dict["3Factor_Monthly"]
    x_axis = st.add_constant(x_axis_data[["Mkt-RF"]])

    '''
    model_output structure
    Key: str = portfolio -> portfolio name
    Value: nested list = [[lambdas],[t-stats]] -> 2nd stage regression
    '''
    model_output = {}

    # Part A
    for portfolio in ports_to_test:
        y_axis_data: pd.DataFrame = pk_dict[portfolio]

        betas = {}
        avg_excess_returns = {}

        for y_axis in list(y_axis_data.columns.values)[1:]:
            model = st.OLS(y_axis_data[y_axis].astype(float), x_axis.astype(float)).fit()
            betas[y_axis] = model.params

            avg_excess_return = y_axis_data[y_axis].astype(float).mean() - x_axis_data.iloc[:, 4].astype(float).mean()
            avg_excess_returns[y_axis] = avg_excess_return

        avg_excess_returns_df = pd.DataFrame(list(avg_excess_returns.values()), columns=["AvgExcessReturn"])
        betas_df = pd.DataFrame(betas).T.drop('const', axis=1)

        model_2 = st.OLS(np.array(avg_excess_returns_df), np.array(st.add_constant(betas_df))).fit()

        model_output[portfolio] = [list(model_2.params), list(model_2.tvalues)]
'''
    # Part B
    for portfolio in ports_to_test:
        x_axis_data: pd.DataFrame = pk_dict["3Factor_Daily"]
        x_axis = st.add_constant(x_axis_data[["Mkt-RF"]])

        date_range = x_axis_data["Date"]
        start_date_locs = [0]
        cur_start_date = "19630701"
        previous_date = ""
        for date in date_range:
            if int(date[:4]) == int(cur_start_date[:4]) + 1:
                if int(date[5:6]) 
'''
    return model_output


# 3-Factor Model
def three_factor_model(pk_dict, ports_to_test):
    x_axis_data: pd.DataFrame = pk_dict["3Factor_Monthly"]
    x_axis = st.add_constant(x_axis_data[["Mkt-RF", "SMB", "HML"]])

    '''
        model_output structure
        Key: str = portfolio -> portfolio name
        Value: nested list = [[lambdas],[t-stats]] -> 2nd stage regression
        '''
    model_output = {}

    for portfolio in ports_to_test:
        y_axis_data: pd.DataFrame = pk_dict[portfolio]

        betas = {}
        avg_excess_returns = {}

        for y_axis in list(y_axis_data.columns.values)[1:]:
            model = st.OLS(y_axis_data[y_axis].astype(float), x_axis.astype(float)).fit()
            betas[y_axis] = model.params

            avg_excess_return = y_axis_data[y_axis].astype(float).mean() - x_axis_data.iloc[:, 4].astype(float).mean()
            avg_excess_returns[y_axis] = avg_excess_return

        avg_excess_returns_df = pd.DataFrame(list(avg_excess_returns.values()), columns=["AvgExcessReturn"])
        betas_df = pd.DataFrame(betas).T.drop('const', axis=1)

        model_2 = st.OLS(np.array(avg_excess_returns_df), np.array(st.add_constant(betas_df))).fit()

        model_output[portfolio] = [list(model_2.params), list(model_2.tvalues)]

    return model_output


# 5-Factor Model
def five_factor_model(pk_dict, ports_to_test):
    x_axis_data: pd.DataFrame = pk_dict["5Factor_Monthly"]
    x_axis = st.add_constant(x_axis_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])

    '''
    model_output structure
    Key: str = portfolio -> portfolio name
    Value: nested list = [[lambdas],[t-stats]] -> 2nd stage regression
    '''
    model_output = {}

    for portfolio in ports_to_test:
        y_axis_data: pd.DataFrame = pk_dict[portfolio]

        betas = {}
        avg_excess_returns = {}

        for y_axis in list(y_axis_data.columns.values)[1:]:
            model = st.OLS(y_axis_data[y_axis].astype(float), x_axis.astype(float)).fit()
            betas[y_axis] = model.params

            avg_excess_return = y_axis_data[y_axis].astype(float).mean() - x_axis_data.iloc[:, 4].astype(float).mean()
            avg_excess_returns[y_axis] = avg_excess_return

        avg_excess_returns_df = pd.DataFrame(list(avg_excess_returns.values()), columns=["AvgExcessReturn"])
        betas_df = pd.DataFrame(betas).T.drop('const', axis=1)

        model_2 = st.OLS(np.array(avg_excess_returns_df), np.array(st.add_constant(betas_df))).fit()

        model_output[portfolio] = [list(model_2.params), list(model_2.tvalues)]

    return model_output


# 6-Factor Model
def six_factor_model(pk_dict, ports_to_test):
    x_axis_data: pd.DataFrame = pk_dict["5Factor_Monthly"]
    x_axis = st.add_constant(x_axis_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]])

    '''
    model_output structure
    Key: str = portfolio -> portfolio name
    Value: nested list = [[lambdas],[t-stats]] -> 2nd stage regression
    '''
    model_output = {}

    for portfolio in ports_to_test:
        y_axis_data: pd.DataFrame = pk_dict[portfolio]

        betas = {}
        avg_excess_returns = {}

        for y_axis in list(y_axis_data.columns.values)[1:]:
            model = st.OLS(y_axis_data[y_axis].astype(float), x_axis.astype(float)).fit()
            betas[y_axis] = model.params

            avg_excess_return = y_axis_data[y_axis].astype(float).mean() - x_axis_data.iloc[:, 4].astype(float).mean()
            avg_excess_returns[y_axis] = avg_excess_return

        avg_excess_returns_df = pd.DataFrame(list(avg_excess_returns.values()), columns=["AvgExcessReturn"])
        betas_df = pd.DataFrame(betas).T.drop('const', axis=1)

        model_2 = st.OLS(np.array(avg_excess_returns_df), np.array(st.add_constant(betas_df))).fit()

        model_output[portfolio] = [list(model_2.params), list(model_2.tvalues)]

    return model_output


# Main method, dictates flow of program
def main(config, config_file, config_path, config_data_type, config_param):
    # Check to see if config necessary; if not, check if data.dat exists; if not, config data.dat
    if config:
        config_data(config_file, config_path, config_data_type, config_param)
        main(False, config_file, config_path, config_data_type, config_param)
    elif path.exists(config_file):
        with open("data.dat", "rb") as pickle_file:
            pk_dict = pk.load(pickle_file)
            ports_to_test = ["25Ports_InvBM_Monthly", "25Ports_OpBM_Monthly", "25Ports_OpInv_Monthly",
                             "25Ports_SizeBM_Monthly", "25Ports_SizeInv_Monthly", "25Ports_SizeOp_Monthly",
                             "48IndustryPorts_Monthly"]
            mkt_data = market_model(pk_dict, ports_to_test)
            three_factor_data = three_factor_model(pk_dict, ports_to_test)
            five_factor_data = five_factor_model(pk_dict, ports_to_test)
            six_factor_data = six_factor_model(pk_dict, ports_to_test)
    else:
        config_data(config_file, config_path, config_data_type, config_param)
        main(False, config_file, config_path, config_data_type, config_param)


# Main method and configuration settings
if __name__ == '__main__':
    config = False
    config_file = "data.dat"
    config_path = "./CSVDataSets/"
    config_data_type = ".csv"
    config_param = ("3Factor_Daily", "3Factor_Monthly",
                    "5Factor_Daily", "5Factor_Monthly",
                    "25Ports_InvBM_Daily", "25Ports_InvBM_Monthly",
                    "25Ports_OpBM_Daily", "25Ports_OpBM_Monthly",
                    "25Ports_OpInv_Daily", "25Ports_OpInv_Monthly",
                    "25Ports_SizeBM_Daily", "25Ports_SizeBM_Monthly",
                    "25Ports_SizeInv_Daily", "25Ports_SizeInv_Monthly",
                    "25Ports_SizeOp_Daily", "25Ports_SizeOp_Monthly",
                    "48IndustryPorts_Daily", "48IndustryPorts_Monthly")
    main(config, config_file, config_path, config_data_type, config_param)
