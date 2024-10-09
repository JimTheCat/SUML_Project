import pandas as pd
from matplotlib.pyplot import xticks

# Read game_details.json and convert it to DataFrame
df = pd.read_json('game_details.json')

import matplotlib.pyplot as plt

# Wykonaj wykres rozkładu cen gier w formie wykresu liniowego z dokładnością do 0.1
prices = df['price'].round(1)
prices.value_counts().sort_index().plot()
plt.xlabel('Price')
plt.ylabel('Number of games')
plt.title('Price distribution')
# plt.xticks(range(0, 100, 5))
plt.show()