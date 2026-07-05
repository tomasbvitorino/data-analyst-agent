import pandas as pd
import numpy as np
from faker import Faker
import random

random.seed(7)
np.random.seed(7)
fake = Faker()
Faker.seed(7)

N = 5000
categories = ["Electronics", "Furniture", "Clothing", "Groceries", "Sports", "Toys", "Books"]
regions = ["North", "South", "East", "West", "Central"]
channels = ["Online", "In-Store"]

rows = []
for i in range(N):
    date = fake.date_between(start_date="-2y", end_date="today")
    category = random.choices(categories, weights=[0.2, 0.1, 0.2, 0.25, 0.1, 0.08, 0.07])[0]
    region = random.choice(regions)
    channel = random.choices(channels, weights=[0.65, 0.35])[0]
    unit_price = round(np.random.gamma(2, {"Electronics": 150, "Furniture": 200, "Clothing": 30,
                                            "Groceries": 8, "Sports": 40, "Toys": 20, "Books": 12}[category]), 2)
    quantity = random.randint(1, 6)
    discount_pct = random.choices([0, 5, 10, 15, 20], weights=[0.5, 0.2, 0.15, 0.1, 0.05])[0]
    revenue = round(unit_price * quantity * (1 - discount_pct / 100), 2)
    cost = round(unit_price * quantity * random.uniform(0.5, 0.7), 2)
    profit = round(revenue - cost, 2)
    rows.append([i + 1, date, category, region, channel, unit_price, quantity, discount_pct, revenue, cost, profit])

df = pd.DataFrame(rows, columns=[
    "order_id", "order_date", "category", "region", "channel",
    "unit_price", "quantity", "discount_pct", "revenue", "cost", "profit"
])
df = df.sort_values("order_date").reset_index(drop=True)
df.to_csv("sales_data.csv", index=False)
print(df.shape)
print(df.head())
