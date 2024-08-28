import json
import pandas as pd

class PowerCurve():
    def __init__(self) -> None:

        with open('turbine_data/turbine_power_curves.json', 'r') as json_file:
                self.power_curves = json.load(json_file)
        
        # Current Available wtgs brand+diameter_KW
        # ['MY193_6250', 'MY212_8000', 'MY212_10000', 'AD116_5000', 'E-101_3050', 'E-101_3500', 'E-115_3000', 'E-115_3200', 'E-126_4200', 'E-126_7500', 'E-126_7580', 'E-141_4200', 
        #  'E-53_800', 'E-70_2000', 'E-70_2300', 'E-82_2000', 'E-82_2300', 'E-82_2350', 'E-82_3000', 'E-92_2350', 'E48_800', 'ENO100_2200', 'ENO114_3500', 'ENO126_3500', 'GE100_2500', 
        #  'GE103_2750', 'GE120_2500', 'GE120_2750', 'GE130_3200', 'MM100_2000', 'MM92_2050', 'N100_2500', 'N117_2400', 'N131_3000', 'N131_3300', 'N131_3600', 'N90_2500', 'S104_3400', 
        #  'S114_3200', 'S114_3400', 'S122_3000', 'S122_3200', 'S126_6150', 'S152_6330', 'SCD168_8000', 'SWT113_2300', 'SWT113_3200', 'SWT120_3600', 'SWT130_3300', 'SWT130_3600', 
        #  'SWT142_3150', 'V100_1800', 'V100_1800_GS', 'V112_3000', 'V112_3075', 'V112_3300', 'V112_3450', 'V117_3300', 'V117_3450', 'V117_3600', 'V126_3000', 'V126_3300', 'V126_3450', 
        #  'V164_8000', 'V164_9500', 'V80_2000', 'V90_2000', 'V90_2000_GS', 'V90_3000', 'VS112_2500']


    def get_power_curve(self, wtg='MY212_8000'):
        
        return pd.DataFrame(self.power_curves[wtg])
    
    def get_wtgs(self):
         
         return list(self.power_curves.keys())


