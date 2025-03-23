import csv
from datetime import datetime
import time
import numpy as np
from gpiozero import MCP3202

       

if __name__ == '__main__':
    try:        

        sample_counter = 0
        voltage_data = []
        pot = MCP3202(channel=0)
        print("-----Starting data acquisition-----")
        start_time = time.time()

        while True:
            sample_counter += 1
            voltage = pot.value
            voltage_data.append(voltage)

    except KeyboardInterrupt:
        print("-----Program interrupted by user-----")

    finally:
        # Signal the acquisition thread to stop
        elapsed_time = time.time() - start_time
        print("-----Data acquisition stopped-----")
        print(f"Time elapsed: {elapsed_time:.5f} seconds")
        print(f"Sample rate: {sample_counter / elapsed_time:.5f} samples per second")
        

        # Generate a unique filename
        filename = datetime.now().strftime("data/vibration_data_%Y%m%d_%H%M%S.csv")
        
        # Write data to CSV file
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            print(type(voltage_data[0]))


            writer.writerow(voltage_data)
    