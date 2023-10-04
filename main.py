
# Imports
import pickle as pk
import os.path as path
import pandas as pd


# Configure dictionary of dataframes for data, convert to pickle object
def config_data(config_file, config_path, config_data_type, config_param):
    pk_dict = {}

    for name in config_param:
        pk_dict[name] = pd.read_csv(config_path + name + config_data_type, dtype=object)

    with open(config_file, "wb") as pk_file:
        pk.dump(pk_dict, pk_file)


# Main method, dictates flow of program
def main(config, config_file, config_path, config_data_type, config_param):
    if config:
        config_data(config_file, config_path, config_data_type, config_param)
    elif path.exists(config_file):
        pass
    else:
        config_data(config_file, config_path, config_data_type, config_param)


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
                    "25Ports_SizeBM_Daily", "25Ports_SizeBM_Monthly",
                    "25Ports_SizeInv_Daily", "25Ports_SizeInv_Monthly",
                    "25Ports_SizeOp_Daily", "25Ports_SizeOp_Monthly",
                    "48IndustryPorts_Daily", "48IndustryPorts_Monthly",
                    "Momentum_Daily", "Momentum_Monthly")
    main(config, config_file, config_path, config_data_type, config_param)


