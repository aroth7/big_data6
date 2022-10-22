# %%
import csv
import math
import numpy as np

# %%
f = open('demand_monopoly.csv', 'r')

reader = csv.DictReader(f)

quantity  = list()
price  = list()

for obs in reader:
  try:
    p  = float( obs['p'] )
  except ValueError:
    p  = 0

  try:
    q  = float( obs['s'] )
  except ValueError:
    q  = 0
  
  quantity.append(q)
  price.append(p)

f.close()

# %%
quantity  = np.array(quantity)
price  = np.array(price)

log_q = np.log(quantity)
log_p = np.log(price)

print("OLS Regression: ", 
      np.linalg.lstsq(np.vstack([log_q, np.ones(len(quantity))]).T, 
                      log_p))
# %%
