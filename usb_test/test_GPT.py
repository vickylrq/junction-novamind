import numpy as np
import re

def extract_numbers(input_string):
    # Use regular expression to find all occurrences of patterns like '=number'
    numbers = re.findall(r'=-?[\d]+', input_string)

    # Remove the '=' symbol and convert to integers
    numbers = [int(num.replace('=', '')) for num in numbers]

    return np.array(numbers)

# Example usage
input_string = "ACC_X=-11ACC_Y=-47ACC_Z=1014GYR_X=-1GYR_Y=-12GYR_Z=14MAG_X=1MAG_Y=-4MAG_Z=-15"
output_array = extract_numbers(input_string)
print(output_array)