import pandas as pd
from pathlib import Path
p=Path('data/raw/Train_data.csv')
if not p.exists():
    p=Path('../data/raw/Train_data.csv')
df=pd.read_csv(p)
out=Path('samples')
out.mkdir(parents=True,exist_ok=True)
sample=df.sample(100, random_state=42).drop(columns=['class'])
sample.to_csv(out/'demo_input.csv', index=False)
print('wrote', out/'demo_input.csv')
