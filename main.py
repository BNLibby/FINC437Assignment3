
# Imports
import pickle as pk
import os.path as path
import pandas as pd
import statsmodels.api as st

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
def market_model(pk_dict):
    pass


# 3-Factor Model
def three_factor_model(pk_dict):
    output_dict = {}

    x_axis_data: pd.DataFrame = pk_dict["3Factor_Monthly"]
    y_axis_data: pd.DataFrame = pk_dict["25Ports_SizeBM_Monthly"]

    x_axis = st.add_constant(x_axis_data[["Mkt-RF", "SMB", "HML"]])
    for y_axis in list(y_axis_data.columns.values)[1:]:
        model = st.OLS(y_axis_data[y_axis].astype(float), x_axis.astype(float)).fit()
        output_dict[y_axis] = [model.params, model.tvalues]


# 5-Factor Model
def five_factor_model(pk_dict):
    pass


# 6-Factor Model
def six_factor_model(pk_dict):
    pass


# Main method, dictates flow of program
def main(config, config_file, config_path, config_data_type, config_param):
    # Check to see if config necessary; if not, check if data.dat exists; if not, config data.dat
    if config:
        config_data(config_file, config_path, config_data_type, config_param)
        main(False, config_file, config_path, config_data_type, config_param)
    elif path.exists(config_file):
        with open("data.dat", "rb") as pickle_file:
            pk_dict = pk.load(pickle_file)
            market_model(pk_dict)
            three_factor_model(pk_dict)
            five_factor_model(pk_dict)
            six_factor_model(pk_dict)
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
                    "5Factor_Daily", "5Factor_monthly",
                    "25Ports_InvBM_Daily", "25Ports_InvBM_Monthly",
                    "25Ports_OpBM_Daily", "25Ports_OpBM_Monthly",
                    "25Ports_OpInv_Daily", "25Ports_OpInv_Monthly",
                    "25Ports_SizeBM_Daily", "25Ports_SizeBM_Monthly",
                    "25Ports_SizeInv_Daily", "25Ports_SizeInv_Monthly",
                    "25Ports_SizeOp_Daily", "25Ports_SizeOp_Monthly",
                    "48IndustryPorts_Daily", "48IndustryPorts_Monthly",
                    "Momentum_Daily", "Momentum_Monthly")
    main(config, config_file, config_path, config_data_type, config_param)


