"""verify_pipeline.py — Verification script v4 (rest/B2B features + model saving)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, kagglehub, os, joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, roc_curve
from xgboost import XGBClassifier
import tensorflow as tf; tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = kagglehub.dataset_download('eoinamoore/historical-nba-data-and-player-box-scores')
games = pd.read_csv(os.path.join(DATA_PATH, 'Games.csv'))
ts = pd.read_csv(os.path.join(DATA_PATH, 'TeamStatistics.csv'))
games = games[games['gameType'] == 'Regular Season'].copy()
ts = ts[ts['gameId'].isin(games['gameId'])].copy()
games['gameDateTimeEst'] = pd.to_datetime(games['gameDateTimeEst'])
games = games.sort_values('gameDateTimeEst').reset_index(drop=True)
ts['gameDateTimeEst'] = pd.to_datetime(ts['gameDateTimeEst'])
ts = ts.sort_values(['gameDateTimeEst', 'gameId']).reset_index(drop=True)

def get_season(dt): return dt.year + 1 if dt.month >= 10 else dt.year
games['season'] = games['gameDateTimeEst'].apply(get_season)
ts['season'] = ts['gameDateTimeEst'].apply(get_season)

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

# One-game shift + Fatigue Features
ts = ts.sort_values(['teamId','gameDateTimeEst']).reset_index(drop=True)
for col in stat_cols: ts[f'{col}_prev'] = ts.groupby('teamId')[col].shift(1)
ts['win_prev'] = ts.groupby('teamId')['win'].shift(1)
ts['prev_game_date'] = ts.groupby('teamId')['gameDateTimeEst'].shift(1)
ts['rest_days'] = (ts['gameDateTimeEst'] - ts['prev_game_date']).dt.days
ts['rest_days'] = ts['rest_days'].clip(upper=7).fillna(7) # Cap at 7 days, Fillna for season openers
ts['b2b'] = (ts['rest_days'] == 1).astype(int)

# Elo
HOME_ADV = 100; K = 20
def compute_elo(gdf):
    gl = gdf[['gameId','gameDateTimeEst','hometeamId','awayteamId','homeScore','awayScore','season']].sort_values('gameDateTimeEst').reset_index(drop=True)
    ss = sorted(gl['season'].unique()); s2w = {s:i//2 for i,s in enumerate(ss)}
    elo,eb,cw = {},{},-1
    for _,g in gl.iterrows():
        w = s2w[g['season']]
        if w != cw: elo = {t:1500 for t in elo}; cw = w
        for t in [g['hometeamId'],g['awayteamId']]:
            if t not in elo: elo[t] = 1500
        eb[(g['gameId'],g['hometeamId'])] = elo[g['hometeamId']]
        eb[(g['gameId'],g['awayteamId'])] = elo[g['awayteamId']]
        if pd.isna(g['homeScore']) or pd.isna(g['awayScore']): continue
        ha = elo[g['hometeamId']]+HOME_ADV
        eh = 1.0/(1.0+10**((elo[g['awayteamId']]-ha)/400))
        hw = 1.0 if g['homeScore']>g['awayScore'] else 0.0
        elo[g['hometeamId']] += K*(hw-eh); elo[g['awayteamId']] += K*((1-hw)-(1-eh))
    return eb
print('Computing Elo...')
elo_map = compute_elo(games)
ts['elo'] = ts.apply(lambda x: elo_map.get((x['gameId'],x['teamId']),np.nan),axis=1)

# Rolling averages
ts['opponentTeamId'] = ts['opponentTeamId'].astype('Int64')
ts['opp_elo'] = ts.apply(lambda x: elo_map.get((x['gameId'],x['opponentTeamId']),1500),axis=1)
WINDOW = 10; prev_cols = [f'{c}_prev' for c in stat_cols]
def wroll(group):
    group = group.sort_values('gameDateTimeEst')
    w = group['win_prev'].fillna(0)*(1+group['opp_elo'].shift(1).fillna(1500)/1500)+0.1
    res = pd.DataFrame(index=group.index)
    for col in prev_cols:
        v = group[col].values.astype(float); ww = w.values.astype(float)
        ra = np.full(len(v), np.nan)
        for i in range(len(v)):
            s = max(0,i-WINDOW+1); sv,sw = v[s:i+1],ww[s:i+1]
            m = np.isfinite(sv)&np.isfinite(sw)
            if m.sum()>0: ra[i] = np.average(sv[m],weights=sw[m])
        res[col.replace('_prev','_roll10')] = ra
    return res
print('Computing rolling averages...')
rr = []
teams = ts['teamId'].unique()
for tid in teams: rr.append(wroll(ts[ts['teamId']==tid].copy()))
ts = ts.join(pd.concat(rr))
roll_cols = [c.replace('_prev','_roll10') for c in prev_cols]

# Game-level feature matrix
home = ts[ts['home']==1].copy(); away = ts[ts['home']==0].copy()
fcols = roll_cols + ['elo', 'rest_days', 'b2b']
hf = home[['gameId','gameDateTimeEst','teamId','season','win']+fcols].rename(columns={c:f'home_{c}' for c in fcols}).rename(columns={'win':'HOME_WIN','teamId':'home_teamId'})
af = away[['gameId','teamId']+fcols].rename(columns={c:f'away_{c}' for c in fcols}).rename(columns={'teamId':'away_teamId'})
gf = hf.merge(af, on='gameId', how='inner')
for c in fcols: gf[f'diff_{c}'] = gf[f'home_{c}']-gf[f'away_{c}']
diff_cols = [f'diff_{c}' for c in fcols]
all_fc = [f'home_{c}' for c in fcols]+[f'away_{c}' for c in fcols]+diff_cols
gf[all_fc] = gf[all_fc].replace([np.inf,-np.inf],np.nan)
gfc = gf.dropna(subset=all_fc+['HOME_WIN'])
print(f'Clean games: {len(gfc):,} | Features: {len(all_fc)} | Home win rate: {gfc["HOME_WIN"].mean():.3f}')

# Train/Test Split
TEST_SEASON = 2025; TRAIN_SEASONS = list(range(2015,2025))
tr = gfc['season'].isin(TRAIN_SEASONS); te = gfc['season']==TEST_SEASON
X_train = gfc.loc[tr,all_fc].values.astype(np.float64); y_train = gfc.loc[tr,'HOME_WIN'].values.astype(int)
X_test = gfc.loc[te,all_fc].values.astype(np.float64); y_test = gfc.loc[te,'HOME_WIN'].values.astype(int)
print(f'Train: {len(X_train):,} | Test: {len(X_test):,}')
scaler = StandardScaler()
Xts = scaler.fit_transform(X_train); Xtes = scaler.transform(X_test)

def go(name, model, is_keras=False):
    print(f'Training {name}...')
    if is_keras:
        model.fit(Xts,y_train,epochs=50,batch_size=32,validation_split=0.2,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)],verbose=0)
        p = model.predict(Xtes,verbose=0).flatten()
    else:
        model.fit(Xts,y_train) if "XGB" not in name else model.fit(Xts[:int(len(Xts)*0.85)], y_train[:int(len(y_train)*0.85)], eval_set=[(Xts[int(len(Xts)*0.85):], y_train[int(len(y_train)*0.85):])], verbose=False)
        p = model.predict_proba(Xtes)[:,1]
    pr = (p>=0.5).astype(int)
    print(f'  {name}: Acc={accuracy_score(y_test,pr):.4f} AUC={roc_auc_score(y_test,p):.4f} Brier={brier_score_loss(y_test,p):.4f}')
    return p, pr

# Models
rf_model = RandomForestClassifier(n_estimators=500,max_depth=10,random_state=42,n_jobs=-1)
rf_p, rf_pr = go('RF', rf_model)
xgb_model = XGBClassifier(learning_rate=0.05,n_estimators=1000,max_depth=6,eval_metric='logloss',early_stopping_rounds=50,random_state=42,use_label_encoder=False,verbosity=0)
xgb_p, xgb_pr = go('XGB', xgb_model)
tf.random.set_seed(42); np.random.seed(42)
mlp = keras.Sequential([layers.Input(shape=(Xts.shape[1],)),layers.Dense(64,activation='relu'),layers.BatchNormalization(),layers.Dropout(0.3),layers.Dense(32,activation='relu'),layers.BatchNormalization(),layers.Dropout(0.2),layers.Dense(16,activation='relu'),layers.Dense(1,activation='sigmoid')])
mlp.compile(optimizer=keras.optimizers.Adam(0.001),loss='binary_crossentropy',metrics=['accuracy'])
mlp_p, mlp_pr = go('MLP', mlp, is_keras=True)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_p, svm_pr = go('SVM', svm_model)

# Save
print('Saving models...')
model_path = 'trained_models.pkl'
joblib.dump({'rf': rf_model, 'xgb': xgb_model, 'svm': svm_model, 'scaler': scaler, 'feature_cols': all_fc}, model_path)
mlp.save('mlp_model.keras')

# Ensemble
ensp = (rf_p+xgb_p+mlp_p+svm_p)/4; enspr = (ensp>=0.5).astype(int)
blp = np.full(len(y_test),y_train.mean()); blpr = np.ones(len(y_test),dtype=int)

def cm(n,yt,yp,ypr): return {'Model':n,'Accuracy':accuracy_score(yt,yp),'ROC-AUC':roc_auc_score(yt,ypr),'Brier Score':brier_score_loss(yt,ypr)}
res = pd.DataFrame([cm('Random Forest',y_test,rf_pr,rf_p),cm('XGBoost',y_test,xgb_pr,xgb_p),cm('MLP Neural Net',y_test,mlp_pr,mlp_p),cm('SVM (RBF)',y_test,svm_pr,svm_p),cm('Ensemble (Soft Vote)',y_test,enspr,ensp),cm('Naive Home Win',y_test,blpr,blp)]).set_index('Model')
print('\n'+'='*65)
print('        MODEL COMPARISON — 2024-25 NBA Season')
print('='*65)
print(res.to_string(float_format='{:.4f}'.format))
print('='*65)
print('\n✓ ALL DONE')
