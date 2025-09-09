#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

people = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']

plt.bar(people, fruit[0], width=0.5, color=colors[0], label=fruit_names[0])
plt.bar(people, fruit[1], width=0.5, color=colors[1], label=fruit_names[1], bottom=fruit[0])
plt.bar(people, fruit[2], width=0.5, color=colors[2], label=fruit_names[2], bottom=fruit[0]+fruit[1])
plt.bar(people, fruit[3], width=0.5, color=colors[3], label=fruit_names[3], bottom=fruit[0]+fruit[1]+fruit[2])

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.ylim(0, 80)
plt.yticks(range(0, 81, 10))
plt.legend()
plt.show()