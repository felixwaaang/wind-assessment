import cdsapi
import xarray as xr
import math

class DataDownloader:
    def __init__(self, input_lat, input_long):
        self.client = cdsapi.Client()
        self.input_lat = input_lat
        self.input_long = input_long
        self.download_data()

    def set_area(self):
        lat_floor = math.floor(self.input_lat)
        lat_ceil = math.ceil(self.input_lat)
        lng_floor = math.floor(self.input_long)
        lng_ceil = math.ceil(self.input_long)

        return [lat_ceil, lng_floor, lat_floor, lng_ceil]

    def download_data(self):
        if self.input_lat is None or self.input_long is None:
            return  # Exit if latitude or longitude is not set

        location_input = f"{self.input_lat}_{self.input_long}"  # Generate location identifier
        year_start = 2000
        year_end = 2023

        for year in range(year_start, year_end + 1):
            self.client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        '100m_u_component_of_wind', '100m_v_component_of_wind', '2m_temperature', 'surface_pressure',
                    ],
                    'year': str(year),
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ],
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'format': 'netcdf',
                    'area': self.set_area()
                },
                f'era5_data/download_{year}.nc'
            )

        nc_files = [f'era5_data/download_{yr}.nc' for yr in range(year_start, year_end + 1)]
        combined_ds = xr.open_mfdataset(nc_files, combine='by_coords')
        combined_ds.to_netcdf(f'era5_data/{location_input}.nc')

