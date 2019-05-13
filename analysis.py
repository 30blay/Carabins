import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sqlalchemy import create_engine
from data_extraction import create_db

db_name = 'data/data.db'
# create_db(db_name)
engine = create_engine('sqlite:///' + db_name)

df = pd.read_sql_query("""SELECT
    height,
    weight,
    position   

    FROM 
    medical NATURAL JOIN fatigue
    
    WHERE height is not null and weight is not null
    """, con=engine.connect())

# Get unique names of species
uniq = list(set(df['position']))

# Set the color map to match the number of species
z = range(1, len(uniq))
hot = plt.get_cmap('hot')
cNorm = colors.Normalize(vmin=0, vmax=len(uniq))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='Paired')

# Plot each species
for i in range(len(uniq)):
    indx = df['position'] == uniq[i]
    plt.scatter(df['weight'][indx], df['height'][indx], color=scalarMap.to_rgba(i), label=uniq[i])

plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Height-Weight correlation')
plt.legend(loc='lower right')
plt.show()
