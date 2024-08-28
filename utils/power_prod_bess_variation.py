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

class power_bess:
    def __init__(self, location, latitude, longitude, target_load, is_ocean):
        self.pattern = r'^[^_]+_([^_]+_[^_]+)_'
        self.available_wtgs = pc.get_wtgs()
        self.selected_wtgs = ['MY193_6250', 'MY212_8000', 'MY212_10000', 'D186_7000', 'V164_9500']
        # self.selected_wtgs = ['MY193_6250', 'MY212_10000']
        self.location = location
        self.lat = latitude
        self.lng = longitude
        self.is_ocean = is_ocean

        self.number_of_wtg = 1 
        self.target_load = target_load 
        self.bess_max = 400000

    def process_data(self):
        input_lat = self.lat
        input_lng = self.lng
        target_load = self.target_load
        number_of_wtg = self.number_of_wtg
        bess_max = self.bess_max
        wtgs = self.selected_wtgs
        is_ocean = self.is_ocean

        output_dir = f'production_output/{self.location}/bess/{self.number_of_wtg}'
        bess_step = int(bess_max / 10)

        for wtg in wtgs:
            data_file = os.path.join(output_dir, f'{number_of_wtg}_{wtg}_{bess_max}_{target_load}.csv')
            storage_sizes = [i for i in range(0, bess_max + 1, bess_step)]
            results = []

            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
            else:
                for size in storage_sizes:
                    nc_file_path = f'era5_data/{self.location}.nc'

                    if not os.path.exists(nc_file_path):
                        raise FileNotFoundError('Data not available. Please run era5_download.py to download the data first.')
                    
                    else:
                        calculator = PowerCalculator(
                            wtg=wtg, 
                            size_of_storage=size, 
                            nc_file=nc_file_path, 
                            input_latitude=input_lat, 
                            input_longitude=input_lng,
                            number_of_WTGs=number_of_wtg, 
                            target_load=target_load,
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

        directory = f'pdf_elements/{self.location}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.save_figure()
        self.get_net_capacity()
        self.get_power_prod_with_multiple_wtgs()
        self.save_figure2()

    def get_net_capacity(self):

        image_path = f'pdf_elements/{self.location}/net_capacity.png'
        if os.path.exists(image_path):
            return
        
        input_lat = self.lat
        input_lng = self.lng
        wtgs = self.selected_wtgs
        nc_file_path = f'era5_data/{self.location}.nc'

        file_path = f'production_output/{self.location}/bess/{self.number_of_wtg}'
        files = os.listdir(file_path)
        wtgs = [re.search(self.pattern, file).group(0)[2:-1] for file in files if re.search(self.pattern, file)]
        print(files)
        print(wtgs)

        rated_outputs = [int(item.split('_')[1]) for item in wtgs]
        i = 0
        net_capacity_factor = []

        for wtg in wtgs:
            calculator = PowerCalculator(
                            wtg=wtg, 
                            size_of_storage=1000000,
                            nc_file=nc_file_path, 
                            input_latitude=input_lat, 
                            input_longitude=input_lng,
                            number_of_WTGs=1, 
                            target_load=10000000000000
                        )
            output, _ = calculator.get_results()
            total_energy = output['gross_production'] * 1000000
            total_yrs = output['years']
            rated_output = rated_outputs[i]
            i += 1
            net_capacity_ratio = (total_energy)/(total_yrs)/(rated_output * 8760)
            net_capacity_factor.append(net_capacity_ratio * 100)

        plt.figure(figsize=(10, 6))
        plt.bar(wtgs, net_capacity_factor)
        plt.xlabel('WTGs')
        plt.ylabel('Gross Capacity Factor (%)')
        plt.title('Gross Capacity Factor by WTGs')
        plt.xticks(rotation=45)  # Rotate x labels if necessary
        plt.tight_layout()  # Adjust layout to fit labels
        plt.savefig(f'pdf_elements/{self.location}/net_capacity.png')
        plt.close()

    def get_power_prod_with_multiple_wtgs(self):
        input_lat = self.lat
        input_lng = self.lng
        wtgs = self.selected_wtgs
        nc_file_path = f'era5_data/{self.location}.nc'
        bess_max = self.bess_max * 0.2
        target_load = self.target_load
        is_ocean = self.is_ocean

        file_path = f'production_output/{self.location}/bess/{self.number_of_wtg}'
        files = os.listdir(file_path)
        wtgs = [re.search(self.pattern, file).group(0)[2:-1] for file in files if re.search(self.pattern, file)]

        for number_of_wtgs in range(2, 7):
            output_path = f'production_output/{self.location}/bess/{number_of_wtgs}'

            for wtg in wtgs:
                print(wtg)
                results = []
                data_file = os.path.join(output_path, f'{number_of_wtgs}_{wtg}_{bess_max}_{target_load}.csv')
                
                if os.path.exists(data_file):
                    continue
                
                else:
                    calculator = PowerCalculator(
                                wtg=wtg, 
                                size_of_storage=bess_max, 
                                nc_file=nc_file_path, 
                                input_latitude=input_lat, 
                                input_longitude=input_lng,
                                number_of_WTGs=number_of_wtgs, 
                                target_load=target_load,
                                is_ocean=is_ocean
                            )
                    output, _ = calculator.get_results()
                    results.append(output)
                    
                    df = pd.DataFrame(results)
                    df['utilised_production_per_year'] = df['utilised_production'] / df['years']
                    df['net_production_per_year'] = df['net_production'] / df['years']

                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    df.to_csv(data_file, index=False)

    def save_figure(self):
        def to_percent(y, _):
            return f'{100 * y:.0f}%'
        
        image_path = f'pdf_elements/{self.location}/power_bess_plots.png'
        if os.path.exists(image_path):
            return

        file_path = f'production_output/{self.location}/bess/{self.number_of_wtg}/'
        files = os.listdir(file_path)
        file_paths = [os.path.join(file_path, file) for file in files if re.search(self.pattern, file) and re.search(self.pattern, file).group(1) in self.selected_wtgs]
        dfs = [pd.read_csv(file) for file in file_paths]

        plt.figure(figsize=(18, 6))
        custom_label = [re.search(self.pattern, file).group(0)[2:-1] for file in files if re.search(self.pattern, file)]

        # Plot Coverage vs bess_storage for each dataset
        plt.subplot(1, 2, 1)
        for i, df in enumerate(dfs):
            plt.plot(df['bess_storage'] / 1000, df['coverage_pct'], marker='o', label=files[i][:-4])
        plt.xlabel('bess_storage (MW)')
        plt.ylabel('Coverage')
        plt.title('Coverage vs BESS Storage')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.legend(custom_label, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

        # Plot Utilised production (GWh) per year vs bess_storage for each dataset
        plt.subplot(1, 2, 2)
        for i, df in enumerate(dfs):
            plt.plot(df['bess_storage'] / 1000, df['coverage_pct'].diff(), marker='o', label=files[i][:-4])
        plt.xlabel('bess_storage (MW)')
        plt.ylabel('Coverage increment')
        plt.title('Coverage increment vs BESS Storage')
        plt.legend(custom_label, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

        # # Plot the power curve
        # plt.subplot(1, 3, 3)
        # for wtg in self.selected_wtgs:
        #     curve = pc.get_power_curve(wtg)
        #     plt.plot(curve['hws'], curve['power'], label=wtg)
        # plt.xlabel('Wind Speed (m/s)')
        # plt.ylabel('Power (kW)')
        # plt.title('Power Curves')
        # plt.legend(custom_label, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

    def save_figure2(self):
        def to_percent(y, _):
            return f'{100 * y:.0f}%'

        image_path = f'pdf_elements/{self.location}/power_wtgs_plots.png'
        if os.path.exists(image_path):
            return

        plt.figure(figsize=(10, 6))

        coverage_pct_values = {label: [] for label in self.selected_wtgs}
        bess_storage_values = {label: [] for label in self.selected_wtgs}

        for num in range(2, 7):
            file_path = f'production_output/{self.location}/bess/{num}/'
            files = os.listdir(file_path)
            
            for file in files:
                match = re.search(self.pattern, file)
                if match and match.group(1) in self.selected_wtgs:
                    file_path_full = os.path.join(file_path, file)
                    df = pd.read_csv(file_path_full)
                    wtg_label = match.group(1)
                    
                    if wtg_label in coverage_pct_values:
                        bess_storage_values[wtg_label].append(num)
                        coverage_pct_values[wtg_label].append(df['coverage_pct'].mean())
                    else:
                        print(f"Warning: WTG label {wtg_label} not found in selected WTGs.")

        # Plotting the data
        for label, coverage in coverage_pct_values.items():
            if coverage:  # Only plot if there is data
                plt.plot(bess_storage_values[label], coverage, marker='o', label=label)

        plt.xlabel('Number of WTGs')
        plt.ylabel('Coverage (%)')
        plt.title('Coverage vs Number of WTGs')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
        # plt.grid(True)

        plt.tight_layout()
        plt.savefig(image_path)

