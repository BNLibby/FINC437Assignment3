
# Imports
import pickle as pk
import os.path as path
import pandas as pd
import statsmodels.api as st
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

# TODO: Create overall regression, see Assignment3Overview.pdf Part C...good luck


class FFModel:
    def __init__(self, dat_file: str, config_path: str, config_file_type: str,
                 factor_fnames: tuple, portfolio_fnames: tuple,
                 ff_factors: dict, start_date: str, window_years: int,
                 step_months: int, time_delimiters: tuple = ("Daily", "Monthly")):
        # Data configuration parameters
        self.dat_file: str = dat_file
        self.config_path: str = config_path
        self.config_file_type: str = config_file_type

        # Pieces Of CSV File Names For Piecemeal Operations In Later Methods
        self.factor_fnames: tuple = factor_fnames
        self.portfolio_fnames: tuple = portfolio_fnames

        # Factors Of Various Models
        '''
        Structure ->
        Key: str == Model Name (ex. Market)
        Value: list == Factors 
        '''
        self.ff_factors: dict = ff_factors

        # Time Parameters For Raw Data
        self.start_date: str = start_date
        self.window_years: int = window_years
        self.step_months: int = step_months

        # Time Delimiter For Differentiation For Daily/Monthly
        '''
        Structure/Uses ->
        is_monthly: bool == True -> Select Monthly Delimiter
        is_monthly: bool == False -> Select Daily Delimiter
        '''
        self.time_delimiters: tuple = time_delimiters

        # Raw Factor Data
        '''
        Structure ->
        Key: str == Full CSV file name (ex. 5Factor_Monthly)
        Value: pd.DataFrame == Full CSV data
        '''
        self.factor_data: dict = {}
        self.portfolio_data: dict = {}

        # Configure Data Before Processing
        self.config_data()

    # Store Data If Needing To Save Memory
    def store_data(self):
        # Store self.factor_data/portfolio_data in self.dat_file
        data: tuple = (self.factor_data, self.portfolio_data)
        with open(self.dat_file, "wb") as outfile:
            pk.dump(data, outfile)

        # Empty self.factor_data/portfolio_data to save space
        self.factor_data = {}
        self.portfolio_data = {}

        return True

    # Retrieve Data For Data Processing
    def retrieve_data(self):
        with open(self.dat_file, "rb") as infile:
            data: tuple = pk.load(infile)
            self.factor_data = data[0]
            self.portfolio_data = data[1]

        return True

    # Configure Data Before Data Processing
    def config_data(self):
        # Configure Daily Data
        for factor in self.factor_fnames:
            for delimiter in self.time_delimiters:
                self.factor_data[factor + delimiter] = pd.read_csv(self.config_path + factor + delimiter
                                                                   + self.config_file_type, dtype=object)
        for portfolio in self.portfolio_fnames:
            for delimiter in self.time_delimiters:
                self.portfolio_data[portfolio + delimiter] = pd.read_csv(self.config_path + portfolio + delimiter
                                                                         + self.config_file_type, dtype=object)

        # Store Data Until Processing Needed
        self.store_data()

        return True

    # Whole Period 2-Stage Regression
    def wp_two_stage(self, model_type: str, is_monthly=True):
        # Retrieve Data From Storage
        self.retrieve_data()

        # Dictionary To Store Model Statistics For Each Portfolio Tested
        '''
        Structure ->
        Key: str = portfolio -> portfolio name
        Value: nested tuple = [[lambdas],[t-stats]] -> 2nd stage regression
        '''
        model_output: dict = {}

        # Store Constant Independent Variables(IV) Once Model Type Determined
        x_axis_data = None
        x_axis = None

        # Determine Model Type And Store Appropriate IVs
        if model_type == "Market" or model_type == "3Factor":
            if is_monthly:
                x_axis_data = self.factor_data["3Factor_Monthly"]
            else:
                x_axis_data = self.factor_data["3Factor_Daily"]
        elif model_type == "5Factor":
            if is_monthly:
                x_axis_data = self.factor_data["5Factor_Monthly"]
                x_axis = st.add_constant(x_axis_data[self.ff_factors[model_type]])
            else:
                x_axis_data = self.factor_data["5Factor_Daily"]
        elif model_type == "6Factor":
            if is_monthly:
                x_axis_data = self.factor_data["5Factor_Monthly"]
            else:
                x_axis_data = self.factor_data["5Factor_Daily"]
        else:
            print("Model Type Not Found")
            return False

        # Finish Adjusting Data Format Before Regression
        x_axis = st.add_constant(x_axis_data[self.ff_factors[model_type]])

        # Perform 2-Stage Regression
        for portfolio_file in self.portfolio_fnames:
            # Store Dependent Variable(DV) Data For Regression
            y_axis_data: pd.DataFrame = self.portfolio_data[portfolio_file + self.time_delimiters[is_monthly]]

            # Stage 1 Regression Statistics
            port_combo_betas = {}
            port_combo_avg_alpha = {}

            # Perform Stage 1 Regression
            for port_combo in tuple(y_axis_data.columns.values)[1:]:
                stage1_regression = st.OLS(y_axis_data[port_combo].astype(float), x_axis.astype(float)).fit()
                port_combo_betas[port_combo] = stage1_regression.params

                avg_alpha = y_axis_data[port_combo].astype(float).mean() - \
                                    x_axis_data.iloc[:, 4].astype(float).mean()
                port_combo_avg_alpha[port_combo] = avg_alpha

            avg_alpha_df = pd.DataFrame(list(port_combo_avg_alpha.values()), columns=["AvgAlpha"])
            betas_df = pd.DataFrame(port_combo_betas).T.drop('const', axis=1)

            stage2_regression = st.OLS(np.array(avg_alpha_df), np.array(st.add_constant(betas_df))).fit()

            model_output[portfolio_file + self.time_delimiters[is_monthly]] = (tuple(stage2_regression.params),
                                                                               tuple(stage2_regression.tvalues))

        # Return Model Output
        return model_output

    # Moving Window 2-Stage Regression
    def mw_two_stage(self, model_type: str, is_monthly=False):
        # Retrieve Data From Storage
        self.retrieve_data()

        # Dictionary To Store Model Statistics For Each Portfolio Tested
        '''
        Structure ->
        Key: str = portfolio -> portfolio name
        Value: nested tuple = [[lambdas],[t-stats]] -> 2nd stage regression
        '''
        model_output: dict = {}

        # Store Constant Independent Variables(IV) Once Model Type Determined
        start_date = datetime.strptime(self.start_date, '%Y%m%d')
        end_date = start_date + relativedelta(years=self.window_years)
        x_axis_data = None
        filtered_x_axis_data = None
        x_axis = None

        # Determine Model Type And Store Appropriate IVs
        if model_type == "Market" or model_type == "3Factor":
            if is_monthly:
                x_axis_data = self.factor_data["3Factor_Monthly"]
            else:
                x_axis_data = self.factor_data["3Factor_Daily"]
        elif model_type == "5Factor":
            if is_monthly:
                x_axis_data = self.factor_data["5Factor_Monthly"]
            else:
                x_axis_data = self.factor_data["5Factor_Daily"]
        elif model_type == "6Factor":
            if is_monthly:
                x_axis_data = self.factor_data["5Factor_Monthly"]
            else:
                x_axis_data = self.factor_data["5Factor_Daily"]
        else:
            print("Model Type Not Found")
            return False

        # Finish Adjusting Data Format Before Regression
        x_axis_data['Date'] = pd.to_datetime(x_axis_data['Date'], format='%Y%m%d')

        # Perform 2-Stage Moving Window Regression
        while end_date <= x_axis_data['Date'].max():
            # Refactor Data Before Performing Regression
            filtered_x_axis_data = x_axis_data[(x_axis_data['Date'] >= start_date) & (x_axis_data['Date'] <= end_date)]
            x_axis = st.add_constant(filtered_x_axis_data[self.ff_factors[model_type]])

            # Perform 2-Stage Regression
            for portfolio_file in self.portfolio_fnames:
                # Store Dependent Variable(DV) Data For Regression
                y_axis_data: pd.DataFrame = self.portfolio_data[portfolio_file + self.time_delimiters[is_monthly]]
                y_axis_data['Date'] = pd.to_datetime(y_axis_data['Date'], format='%Y%m%d')
                filtered_y_data = y_axis_data[(y_axis_data['Date'] >= start_date) & (y_axis_data['Date'] <= end_date)]

                port_combo_betas = {}
                port_combo_avg_alpha = {}
                for port_combo in tuple(filtered_y_data.columns.values)[1:]:
                    stage1_regression = st.OLS(filtered_y_data[port_combo].astype(float), x_axis.astype(float)).fit()
                    port_combo_betas[port_combo] = stage1_regression.params

                    avg_alpha = filtered_y_data[port_combo].astype(float).mean() - \
                                        filtered_x_axis_data.iloc[:, 4].astype(float).mean()
                    port_combo_avg_alpha[port_combo] = avg_alpha

                avg_alpha_df = pd.DataFrame(list(port_combo_avg_alpha.values()), columns=["AvgAlpha"])
                betas_df = pd.DataFrame(port_combo_betas).T.drop('const', axis=1)

                stage2_regression = st.OLS(np.array(avg_alpha_df), np.array(st.add_constant(betas_df))).fit()

                model_output[portfolio_file + self.time_delimiters[is_monthly]] = [list(stage2_regression.params),
                                                                                   list(stage2_regression.tvalues)]

            # Move Window By 1 Month
            start_date += relativedelta(months=self.step_months)
            end_date = start_date + relativedelta(years=self.window_years)

        # Return Model Output
        return model_output


# Main method, dictates flow of program
def main():
    # Define FFModel Parameters
    dat_file: str = "data.dat"
    config_path: str = "./CSVDataSets/"
    config_file_type: str = ".csv"
    factor_fnames: tuple = ("3Factor_", "5Factor_")
    portfolio_fnames: tuple = ("25Ports_InvBM_", "25Ports_OpBM_", "25Ports_OpInv_", "25Ports_SizeBM_",
                               "25Ports_SizeInv_", "25Ports_SizeOp_", "48IndustryPorts_")
    ff_factors: dict = {"Market":   ["Mkt-RF"],
                        "3Factor": ["Mkt-RF", "SMB", "HML"],
                        "5Factor": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
                        "6Factor": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]}
    start_date: str = "19630701"
    window_years: int = 1
    step_months: int = 1
    time_delimiters: tuple = ("Daily", "Monthly")

    # Create FFModel Object
    ff_model: FFModel = FFModel(dat_file, config_path, config_file_type,
                                factor_fnames, portfolio_fnames,
                                ff_factors, start_date, window_years,
                                step_months, time_delimiters)

    # Models To Test
    models_to_test = ["Market", "3Factor", "5Factor", "6Factor"]

    # Part A
    print("^^^^^^^^^^^^^^^^^^^^^^^")
    print("WHOLE PERIOD REGRESSION")
    for model in models_to_test:
        model_output: dict = ff_model.wp_two_stage(model, True)
        print("------------")
        print("Model Type: " + model)
        for key in model_output.keys():
            print(key)
            print("    Lambdas:", end=" ")
            for lambda_val in model_output[key][0]:
                print(f'{lambda_val:.4f}', end=" ; ")
            print()
            print("    T-Values:", end=" ")
            for t_val in model_output[key][1]:
                print(f'{t_val:.4f}', end=" ; ")
            print()

    # Part B
    print("^^^^^^^^^^^^^^^^^^^^^^^")
    print("MOVING WINDOW PROGRESSION")
    for model in models_to_test:
        model_output: dict = ff_model.mw_two_stage(model, False)
        print("------------")
        print("Model Type: " + model)
        for key in model_output.keys():
            print(key)
            print("    Lambdas:", end=" ")
            for lambda_val in model_output[key][0]:
                print(f'{lambda_val:.4f}', end=" ; ")
            print()
            print("    T-Values:", end=" ")
            for t_val in model_output[key][1]:
                print(f'{t_val:.4f}', end=" ; ")
            print()


# Main method and configuration settings
if __name__ == '__main__':
    main()
