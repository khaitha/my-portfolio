import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def hello_world():
    """Prints 'Hello, world!' to the console."""
    try:
        print("Hello, world!")
        return "Success"  # Indicate successful execution
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Failure"  # Indicate failure

if __name__ == "__main__":
    result = hello_world()
    # Basic plotting example using matplotlib
    try:
        x = np.linspace(0, 2 * np.pi, 100)  #Sample x-values
        y = np.sin(x) #Sample y values for sin(x)

        plt.plot(x,y)
        plt.title("Sine Wave")
        plt.xlabel("x")
        plt.ylabel("sin(x)")
        plt.grid(True)
        plt.show()

        # Sample pandas DataFrame
        data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        df = pd.DataFrame(data)
        print(df.describe()) #Descriptive statistics of the pandas DataFrame
        
    except Exception as e:
        print(f"Plotting/Dataframe error: {e}")