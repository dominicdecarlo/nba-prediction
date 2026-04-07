"""predict_today.py — Predicts tonight's NBA games using trained models and latest stats."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, kagglehub, os, joblib
import datetime
from nba_api.live.nba.endpoints import scoreboard

def get_american_odds(prob):
    if prob >= 0.5: return f"-{int(round((prob / (1 - prob)) * 100))}"
    else: return f"+{int(round(((1 - prob) / prob) * 100))}"

# 1. Fetch Tonight's Schedule
print("Fetching tonight's NBA schedule from nba_api...")
board = scoreboard.ScoreBoard()
games_today = board.games.get_dict()

if not games_today:
    print("No games scheduled for today.")
    import sys; sys.exit(0)

print(f"Found {len(games_today)} games for today.")
today_date = pd.Timestamp(datetime.datetime.now().date())

# 2. Load Models
print("Loading trained models...")
model_path = 'trained_models.pkl'
if not os.path.exists(model_path):
    print("Error: trained_models.pkl not found! Run verify_pipeline.py first.")
    import sys; sys.exit(1)
models = joblib.load(model_path)
rf = models['rf']
xgb = models['xgb']
svm = models['svm']
scaler = models['scaler']
all_fc = models['feature_cols']

# Load MLP (Keras)
import tensorflow as tf; tf.get_logger().setLevel('ERROR')
from tensorflow import keras
try:
    mlp = keras.models.load_model('mlp_model.keras')
except:
    mlp = None

# 3. Compute Latest Stats (using historical data)
print("Computing latest team stats and Elo...")
DATA_PATH = kagglehub.dataset_download('eoinamoore/historical-nba-data-and-player-box-scores')
h_games = pd.read_csv(os.path.join(DATA_PATH, 'Games.csv'))
ts = pd.read_csv(os.path.join(DATA_PATH, 'TeamStatistics.csv'))
h_games = h_games[h_games['gameType'] == 'Regular Season'].copy()
ts = ts[ts['gameId'].isin(h_games['gameId'])].copy()
h_games['gameDateTimeEst'] = pd.to_datetime(h_games['gameDateTimeEst'])
h_games = h_games.sort_values('gameDateTimeEst').reset_index(drop=True)
ts['gameDateTimeEst'] = pd.to_datetime(ts['gameDateTimeEst'])
ts = ts.sort_values(['gameDateTimeEst', 'gameId']).reset_index(drop=True)

# Compute basic stats
fga_safe = ts['fieldGoalsAttempted'].replace(0, np.nan)
fgm_safe = ts['fieldGoalsMade'].replace(0, np.nan)
ts['possessions'] = 0.96*(ts['fieldGoalsAttempted']-ts['reboundsOffensive']+ts['turnovers']+0.44*ts['freeThrowsAttempted'])
poss_safe = ts['possessions'].replace(0, np.nan)
ts['eFG_pct'] = (ts['fieldGoalsMade']+0.5*ts['threePointersMade'])/fga_safe
ts['TOV_pct'] = ts['turnovers']/poss_safe
total_reb = (ts['reboundsOffensive']+ts['reboundsDefensive']).replace(0, np.nan)
ts['ORB_pct'] = ts['reboundsOffensive']/total_reb
ts['PPG'] = ts['teamScore'].astype(float)
ts['OppPPG'] = ts['opponentScore'].astype(float)
ts['AST_ratio'] = ts['assists']/fgm_safe
ts['ThreePT_rate'] = ts['threePointersMade']/fga_safe
ts['pace'] = ts['possessions']
ts['net_rating'] = (ts['teamScore']-ts['opponentScore'])/poss_safe*100

opp_stats = ts[['gameId','teamId','eFG_pct','TOV_pct']].rename(columns={'teamId':'opponentTeamId','eFG_pct':'def_eFG_pct','TOV_pct':'def_TOV_pct'})
ts = ts.merge(opp_stats, on=['gameId','opponentTeamId'], how='left')

stat_cols = ['eFG_pct','TOV_pct','ORB_pct','PPG','OppPPG','AST_ratio','ThreePT_rate','def_eFG_pct','def_TOV_pct','pace','net_rating','possessions']
for col in stat_cols: ts[col] = ts[col].replace([np.inf, -np.inf], np.nan)
ts = ts.sort_values(['teamId','gameDateTimeEst']).reset_index(drop=True)

# Compute final Elo (without window resets for simplicity here, or use 2-season reset if we process all)
HOME_ADV = 100; K = 20
def get_season(dt): return dt.year + 1 if dt.month >= 10 else dt.year
h_games['season'] = h_games['gameDateTimeEst'].apply(get_season)
ss = sorted(h_games['season'].unique()); s2w = {s:s for s in ss}
elo = {}; cw = -1
for _,g in h_games.iterrows():
    w = s2w[g['season']]
    if w != cw:
        for t in elo: elo[t] = elo[t] * 0.8 + 1500 * 0.2
        cw = w
    for t in [g['hometeamId'],g['awayteamId']]:
        if t not in elo: elo[t] = 1500
    if pd.isna(g['homeScore']): continue
    
    ha = elo[g['hometeamId']]+HOME_ADV
    eh = 1.0/(1.0+10**((elo[g['awayteamId']]-ha)/400))
    hw = 1.0 if g['homeScore']>g['awayScore'] else 0.0
    
    mov = max(1, abs(g['homeScore'] - g['awayScore']))
    winner_elo = elo[g['hometeamId']] + HOME_ADV if hw == 1.0 else elo[g['awayteamId']]
    loser_elo = elo[g['awayteamId']] if hw == 1.0 else elo[g['hometeamId']] + HOME_ADV
    elo_diff = winner_elo - loser_elo
    
    mov_ratio = min(1.0, (mov - 1) / 29.0)
    multiplier = (1.0 + mov_ratio) * max(0.1, 1.0 - (elo_diff / 500.0))
    multiplier = min(2.0, multiplier)
    
    elo[g['hometeamId']] += K * multiplier * (hw - eh)
    elo[g['awayteamId']] += K * multiplier * ((1 - hw) - (1 - eh))

# Map team names to IDs (Approximation using known mappings from API to Kaggle dataset)
# Using a name to ID dictionary to match API teams to historical IDs
team_name_to_id = {}
for _, row in ts.groupby('teamId').last().iterrows():
    # Kaggle dataset has team names like "Boston Celtics" maybe? Actually it uses teamName if available
    # We will just map it dynamically
    pass

# We need the official team mapping
# NBA API team properties
from nba_api.stats.static import teams
nba_teams = teams.get_teams()
name_to_id = {t['city'] + " " + t['nickname']: t['id'] for t in nba_teams}
name_to_id['LA Clippers'] = name_to_id['Los Angeles Clippers']

# Get latest rolling stats per team
# We compute a 10-game window up to their last game in the dataset
WINDOW = 10
latest_stats = {}
for tid in ts['teamId'].unique():
    group = ts[ts['teamId']==tid].tail(WINDOW)
    # Win weights
    w = group['win'].fillna(0) + 0.1 # Simplified recent weight since we don't need exact opp_elo here
    stat_dict = {}
    for col in stat_cols:
        v = group[col].values.astype(float)
        ww = w.values.astype(float)
        m = np.isfinite(v)&np.isfinite(ww)
        stat_dict[f'{col}_roll10'] = np.average(v[m], weights=ww[m]) if m.sum() > 0 else 0.0
    
    # Rest and Fatigue
    last_game_date = group['gameDateTimeEst'].iloc[-1]
    rest_days = (today_date - last_game_date).days
    rest_days = min(rest_days, 7)
    b2b = 1 if rest_days == 1 else 0
    
    stat_dict['elo'] = elo.get(tid, 1500)
    stat_dict['rest_days'] = rest_days
    stat_dict['b2b'] = b2b
    
    latest_stats[tid] = stat_dict

print("\n" + "="*65)
print("        TONIGHT's PREDICTIONS (HOME WIN % & ODDS)")
print("="*65)

# 4. Predict
rows_to_predict = []
game_names = []

for game in games_today:
    home_name = game['homeTeam']['teamCity'] + " " + game['homeTeam']['teamName']
    away_name = game['awayTeam']['teamCity'] + " " + game['awayTeam']['teamName']
    
    if home_name not in name_to_id or away_name not in name_to_id:
        continue
    
    hid = name_to_id[home_name]
    aid = name_to_id[away_name]
    
    if hid not in latest_stats or aid not in latest_stats:
        continue
        
    hs = latest_stats[hid]
    ast = latest_stats[aid]
    
    f_dict = {}
    for c in stat_cols:
        f_dict[f'home_{c}_roll10'] = hs[f'{c}_roll10']
        f_dict[f'away_{c}_roll10'] = ast[f'{c}_roll10']
        f_dict[f'diff_{c}_roll10'] = hs[f'{c}_roll10'] - ast[f'{c}_roll10']
    
    f_dict['home_elo'] = hs['elo']
    f_dict['away_elo'] = ast['elo']
    f_dict['diff_elo'] = hs['elo'] - ast['elo']
    
    f_dict['home_rest_days'] = hs['rest_days']
    f_dict['away_rest_days'] = ast['rest_days']
    f_dict['diff_rest_days'] = hs['rest_days'] - ast['rest_days']
    
    f_dict['home_b2b'] = hs['b2b']
    f_dict['away_b2b'] = ast['b2b']
    f_dict['diff_b2b'] = hs['b2b'] - ast['b2b']
    
    # Ensure columns match all_fc order
    row = [f_dict.get(c, 0.0) for c in all_fc]
    rows_to_predict.append(row)
    game_names.append(f"{away_name} @ {home_name}")

if not rows_to_predict:
    print("No matches could be mapped to historical dataset.")
    import sys; sys.exit(0)

X = np.array(rows_to_predict, dtype=np.float64)
# Scale
X_scaled = scaler.transform(X)

# Probabilities
rf_probs = rf.predict_proba(X_scaled)[:, 1]
xgb_probs = xgb.predict_proba(X_scaled)[:, 1]
svm_probs = svm.predict_proba(X_scaled)[:, 1]

if mlp:
    mlp_probs = mlp.predict(X_scaled, verbose=0).flatten()
    ens_probs = (rf_probs + xgb_probs + svm_probs + mlp_probs) / 4
else:
    ens_probs = (rf_probs + xgb_probs + svm_probs) / 3

for i, g in enumerate(game_names):
    p = ens_probs[i]
    odds = get_american_odds(p)
    print(f"{g.ljust(35)} | Home Win: {p*100:5.1f}% | Fair Odds: {odds}")

print("="*65)
