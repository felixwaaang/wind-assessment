import cdsapi
import xarray as xr
import math
import caffeine
import os

caffeine.on(display=False)
location_input = 'saudi'

dataset = "reanalysis-era5-single-levels"
request = {
    'product_type': ['reanalysis'],
    'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure'],
    'year': ['2022', '2023', '2024'],
    'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
    'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
    'data_format': 'netcdf',
    'download_format': 'unarchived',
    'area': [-22, 15, -23, 16]
}

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()
            
# nc_files = [f'era5_data/{location_input}_{year}.nc' for year in range(year_start, year_end + 1)]
nc_files = ['era5_data/saudi_2019-21.nc', 'era5_data/saudi_2022-24.nc']
combined_ds = xr.open_mfdataset(nc_files, combine='by_coords')
combined_ds.to_netcdf(f'era5_data/{location_input}.nc')

caffeine.off()

