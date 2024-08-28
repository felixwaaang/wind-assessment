from utils.bilinear_interpolate import BilinearInterpolator
from utils.power_curve import PowerCurve
from datetime import datetime
import scipy.special
import netCDF4 as nc
import numpy as np
import pandas as pd
import re

class PowerCalculator():
    def __init__(self, wtg, number_of_WTGs, target_load, size_of_storage, input_longitude, input_latitude, nc_file, 
                 deterministic_loss=0.972, stochastic_loss=0.946, shape=3.00, height=100, is_ocean=False):
        
        self.interpolator = BilinearInterpolator()
        self.power_curve = PowerCurve()
        self.df_power_curve = self.power_curve.get_power_curve(wtg=wtg)
        self.is_ocean = is_ocean
        
        self.number_of_WTGs = number_of_WTGs
        self.target_load = target_load
        self.size_of_storage = size_of_storage
        self.deterministic_loss = deterministic_loss
        self.stochastic_loss = stochastic_loss
        self.shape = shape
        self.input_longitude = input_longitude
        self.input_latitude = input_latitude
        self.height = height
        self.capacity = float(re.search(r'_(\d+)', wtg).group(1))
        
        self.nc_file = nc_file
        self.ds = nc.Dataset(nc_file)
        self.time_var = self.ds.variables['valid_time']
        self.time_units = self.time_var.units
        self.time_calendar = self.time_var.calendar

        self.time_values = self.time_var[:]
        self.times = nc.num2date(self.time_values, units=self.time_units, calendar=self.time_calendar)
        
        self.longitude_list = self.ds.variables['longitude'][:].tolist()
        self.latitude_list = self.ds.variables['latitude'][:].tolist()
        self.times_list = [time.strftime('%Y-%m-%d %H:00:00') for time in self.times]
        
        self.bounding_box = self.interpolator.find_bounding_box(self.input_latitude, self.input_longitude, self.latitude_list, self.longitude_list)
        self.bounding_box_dfs = [self.interpolator.get_df(self.ds, pair[0], pair[1], self.latitude_list, self.longitude_list, self.times_list, self.is_ocean) for pair in self.bounding_box]
        
        self.df = self._initialize_df()
        self._process_data()

    def _initialize_df(self):
        hws_interpolated = self.interpolator.bilinear_interpolation(
            self.input_latitude, self.input_longitude, 
            self.bounding_box[0][0], self.bounding_box[0][1], 
            self.bounding_box[3][0], self.bounding_box[3][1], 
            self.bounding_box_dfs[0]['wind_velocity'].to_list(), 
            self.bounding_box_dfs[2]['wind_velocity'].to_list(), 
            self.bounding_box_dfs[1]['wind_velocity'].to_list(), 
            self.bounding_box_dfs[3]['wind_velocity'].to_list()
        )

        sp_interpolated = self.interpolator.bilinear_interpolation(
            self.input_latitude, self.input_longitude, 
            self.bounding_box[0][0], self.bounding_box[0][1], 
            self.bounding_box[3][0], self.bounding_box[3][1], 
            self.bounding_box_dfs[0]['sp'].to_list(), 
            self.bounding_box_dfs[2]['sp'].to_list(), 
            self.bounding_box_dfs[1]['sp'].to_list(), 
            self.bounding_box_dfs[3]['sp'].to_list()
        )

        temp_interpolated = self.interpolator.bilinear_interpolation(
            self.input_latitude, self.input_longitude, 
            self.bounding_box[0][0], self.bounding_box[0][1], 
            self.bounding_box[3][0], self.bounding_box[3][1], 
            self.bounding_box_dfs[0]['temperature'].to_list(), 
            self.bounding_box_dfs[2]['temperature'].to_list(), 
            self.bounding_box_dfs[1]['temperature'].to_list(), 
            self.bounding_box_dfs[3]['temperature'].to_list()
        )

        assert len(hws_interpolated) == len(sp_interpolated) == len(temp_interpolated), "Lists must be of the same length"
        print(hws_interpolated[:10])

        #air density correction
        hws_density_corrected = [hws * ((sp / (287.058 * temp))/1.225)**(1/3) for hws, sp, temp in zip(hws_interpolated, sp_interpolated, temp_interpolated)]
        print(hws_density_corrected[:10])

        #height correction to 150m
        if self.is_ocean is True:
            hws_density_corrected = [hws * (self.height/100)**0.1 for hws in hws_density_corrected] #0.143 if on land
            print(hws_density_corrected[:10])
        else:
            hws_density_corrected = [hws * (self.height/100)**0.143 for hws in hws_density_corrected] 

        df = pd.DataFrame({'hws': hws_density_corrected}, index=self.times_list)
        df.index = pd.to_datetime(df.index)
        df['Day'] = df.index.date
        df['DayIndex'] = df['Day'].astype('category').cat.codes + 1
        df = df.drop(columns=['Day'])
        return df
    
    def _process_data(self):
        df = self.df
        match_positions = [(self.df_power_curve['hws'] <= item).idxmin() + 1 for item in df['hws'].values]
        
        df['match'] = match_positions
        df['lower_u'] = self.df_power_curve['hws'][df['match'] - 2].tolist()
        df['upper_u'] = self.df_power_curve['hws'][df['match'] - 1].tolist()
        df['lower_p'] = self.df_power_curve['power'][df['match'] - 2].tolist()
        df['upper_p'] = self.df_power_curve['power'][df['match'] - 1].tolist()

        hours = df['DayIndex'].iloc[-1] * 24
        requirement = self.target_load * hours / 1000000  # GWh
        scale = (1 - self.stochastic_loss) / scipy.special.gamma(1 + 1 / self.shape)

        df['gross_p'] = self.number_of_WTGs * (df['lower_p'] + (df['upper_p'] - df['lower_p']) * (df['hws'] - df['lower_u']) / (df['upper_u'] - df['lower_u']))
        df['losses'] = [self.calculate_loss(self.deterministic_loss, scale, self.shape) for _ in range(df.shape[0])]
        df['net_p'] = df['losses'] * df['gross_p']
        df['curtailed'] = df.apply(lambda row: self.target_load if row['net_p'] > self.target_load else row['net_p'], axis=1)
        df['excess'] = df.apply(lambda row: (row['net_p'] - self.target_load) if row['net_p'] > self.target_load else 0, axis=1)
        df['shortfall'] = df.apply(lambda row: (self.target_load - row['net_p']) if row['net_p'] < self.target_load else 0, axis=1)

        df['available'] = 0
        df['stored'] = 0
        df.iloc[0, df.columns.get_loc('available')] = int(df.iloc[0]['shortfall']) if df.iloc[0]['shortfall'] < df.iloc[0]['stored'] else int(df.iloc[0]['stored'])
        df.iloc[0, df.columns.get_loc('stored')] = self.size_of_storage if df['excess'].iloc[0] > self.size_of_storage else int(df['excess'].iloc[0])

        prev_stored = df.iloc[0, df.columns.get_loc('stored')]
        for i, (row, frame) in enumerate(df.iloc[1:].iterrows(), start=1):
            excess_val = int(frame['excess'])
            shortfall_val = int(frame['shortfall'])
            available_val = shortfall_val if shortfall_val < prev_stored else prev_stored

            df.iloc[i, df.columns.get_loc('available')] = available_val
            df.iloc[i, df.columns.get_loc('stored')] = self.size_of_storage if prev_stored + excess_val - available_val > self.size_of_storage else prev_stored + excess_val - available_val

            prev_stored = df.iloc[i, df.columns.get_loc('stored')]

        df['net_plus_stored'] = df['curtailed'] + df['available']
        df['coverage_pct'] = df['net_plus_stored'] / self.target_load
        df['storage_pct'] = df['stored'] / self.size_of_storage

        self.df = df
        self.hours = hours
        self.requirement = requirement
    
    def calculate_loss(self, deterministic_loss, scale, shape):
        return deterministic_loss * (1 - scale * (-np.log(np.random.rand()))**(1 / shape))

    def get_results(self):
        df = self.df
        net_production = df['net_p'].sum() / 1000000  # GWh
        net_plus_stored = df['net_plus_stored'].sum() / 1000000  # GWh
        net_production_ratio = net_production / self.requirement
        average_u = df['hws'].mean()
        years = self.hours / 8766
        gross = df['gross_p'].sum() / 1000000  # GWh

        results = {
            'bess_storage': self.size_of_storage,
            'height': self.height,
            "net_production": net_production, #Possible Net Production in GWh
            "utilised_production": net_plus_stored, #Utilised Production (Net production plus stored) GWh
            'coverage_pct': net_plus_stored / (self.requirement),
            "gross_production": gross, # Possible Gross production in GWh
            "net_p_ratio": net_production_ratio, #Net production ratio in percentage
            "average_hws": average_u,
            "capacity_pct": net_production / (self.capacity * self.hours),
            "years": years,
        }

        return results, self.df #remember i added a self.df here

    def get_formatted_df(self):
        formats = {
            'hws': '{:.2f}',
            'lower_u': '{:.0f}',
            'upper_u': '{:.0f}',
            'gross_p': '{:.2f}',
            'losses': '{:.2%}',
            'excess': '{:.2f}',
            'net_p': '{:.2f}',
            'stored': '{:.2f}',
            'coverage_pct': '{:.2%}',
            'storage_pct': '{:.2%}'
        }

        formatted_df = self.df.copy()
        for col, fmt in formats.items():
            formatted_df[col] = self.df[col].map(fmt.format)

        return formatted_df