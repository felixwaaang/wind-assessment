from utils.calculate_power_production import PowerCalculator
from utils.power_curve import PowerCurve
from utils.era5_download import DataDownloader
import scipy.special
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from matplotlib.ticker import FuncFormatter

pc = PowerCurve()

class power_height:
    def __init__(self, location, latitude, longitude, target_load, is_ocean):
        self.pattern = r'\d+_([A-Z]+\d+_\d+)'
        self.available_wtgs = pc.get_wtgs()
        self.selected_wtgs = ['MY193_6250', 'D186_7000', 'MY212_8000', 'MY212_10000']
        # self.selected_wtgs = ['MY193_6250', 'MY212_10000']
        self.location = location
        self.lat = latitude
        self.lng = longitude
        self.is_ocean = is_ocean

        self.target_load = target_load
        self.number_of_wtg = 1

    def process_data(self):
        input_lat = self.lat
        input_lng = self.lng
        target_load = self.target_load
        number_of_wtg = self.number_of_wtg
        wtgs = self.selected_wtgs
        is_ocean = self.is_ocean

        output_dir = f'production_output/{self.location}/height/{self.number_of_wtg}'

        for wtg in wtgs:
            data_file = os.path.join(output_dir, f'{number_of_wtg}_{wtg}_{target_load}_height.csv')
            heights = [i for i in range(80, 161, 5)]
            results = []

            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
            else:
                for height in heights:
                    nc_file_path = f'era5_data/{self.location}.nc'

                    if not os.path.exists(nc_file_path):
                        downloader = DataDownloader()
                        downloader.download_data(location_input=self.location)
                    else:
                        calculator = PowerCalculator(
                            wtg=wtg, 
                            size_of_storage=0, 
                            nc_file=nc_file_path, 
                            input_latitude=input_lat, 
                            input_longitude=input_lng,
                            number_of_WTGs=number_of_wtg, 
                            target_load=target_load,
                            height=height,
                            is_ocean=is_ocean
                        )
                        output, _ = calculator.get_results()
                        results.append(output)
                
                df = pd.DataFrame(results)
                df['utilised_production_per_year'] = df['utilised_production'] / df['years']
                df['net_production_per_year'] = df['net_production'] / df['years']

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                df.to_csv(data_file, index=False)

        self.save_figure()

    def save_figure(self):
        wtgs = self.selected_wtgs

        def to_percent(y, _):
            return f'{100 * y:.3f}%'

        file_path = f'production_output/{self.location}/height/{self.number_of_wtg}'
        files = os.listdir(file_path)
        file_paths = [os.path.join(file_path, file) for file in files if (match := re.search(self.pattern, file)) and match.group(1) in wtgs]   
        dfs = [pd.read_csv(file) for file in file_paths]

        plt.figure(figsize=(18, 6))
        custom_label = [re.search(self.pattern, file).group(1) for file in files if re.search(self.pattern, file)]

        # Plot Coverage vs height for each dataset
        plt.subplot(1, 2, 1)
        for i, df in enumerate(dfs):
            plt.plot(df['height'], df['coverage_pct'], marker='o', label=files[i][:-4])
        plt.xlabel('Height (m)')
        plt.ylabel('Coverage')
        plt.title('Coverage vs Height')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.ylim(df['coverage_pct'].min()-0.05, plt.ylim()[1])
        plt.legend(custom_label,loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

        # Plot Utilised production (GWh) per year vs height for each dataset
        plt.subplot(1, 2, 2)
        for i, df in enumerate(dfs):
            plt.plot(df['height'], df['coverage_pct'].diff(), marker='o', label=files[i][:-4])
        plt.xlabel('Height (m)')
        plt.ylabel('Coverage increment')
        plt.title('Coverage increment vs Height')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.legend(custom_label,loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

        plt.tight_layout()
        plt.savefig(f'pdf_elements/{self.location}/power_height_plots.png')
        plt.close()
