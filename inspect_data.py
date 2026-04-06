import kagglehub, os, pandas as pd

path = kagglehub.dataset_download('eoinamoore/historical-nba-data-and-player-box-scores')
print('PATH:', path)
print()

for f in sorted(os.listdir(path)):
    fp = os.path.join(path, f)
    sz = os.path.getsize(fp)
    print(f'  {f} ({sz:,} bytes)')
print()

# games.csv
games = pd.read_csv(os.path.join(path, 'games.csv'), nrows=5)
print('=== games.csv ===')
print('Columns:', list(games.columns))
print()
print(games.iloc[0])
print()

# team_statistics files
for fn in sorted(os.listdir(path)):
    if 'team' in fn.lower() or 'Team' in fn:
        df = pd.read_csv(os.path.join(path, fn), nrows=3)
        print(f'=== {fn} ===')
        print('Columns:', list(df.columns))
        print()
        print(df.iloc[0])
        print()
