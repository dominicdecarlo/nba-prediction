import json

nb_path = 'c:\\Users\\jackf\\.gemini\\antigravity\\scratch\\nba-prediction\\nba_prediction.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

old_block = """    all_seasons = sorted(gl['season'].unique())
    s2w = {s: i // 2 for i, s in enumerate(all_seasons)}
    elo, eb, cw = {}, {}, -1

    for _, g in gl.iterrows():
        w = s2w[g['season']]
        if w != cw:
            elo = {t: 1500 for t in elo}; cw = w
        for t in [g['hometeamId'], g['awayteamId']]:
            if t not in elo: elo[t] = 1500
        eb[(g['gameId'], g['hometeamId'])] = elo[g['hometeamId']]
        eb[(g['gameId'], g['awayteamId'])] = elo[g['awayteamId']]
        if pd.isna(g['homeScore']) or pd.isna(g['awayScore']): continue
        ha = elo[g['hometeamId']] + HOME_ADVANTAGE
        eh = 1.0 / (1.0 + 10 ** ((elo[g['awayteamId']] - ha) / 400))
        hw = 1.0 if g['homeScore'] > g['awayScore'] else 0.0
        elo[g['hometeamId']] += K_FACTOR * (hw - eh)
        elo[g['awayteamId']] += K_FACTOR * ((1 - hw) - (1 - eh))"""

new_block = """    all_seasons = sorted(gl['season'].unique())
    s2w = {s: s for s in all_seasons}
    elo, eb, cw = {}, {}, -1

    for _, g in gl.iterrows():
        w = s2w[g['season']]
        if w != cw:
            for t in elo: elo[t] = elo[t] * 0.8 + 1500 * 0.2
            cw = w
        for t in [g['hometeamId'], g['awayteamId']]:
            if t not in elo: elo[t] = 1500
        eb[(g['gameId'], g['hometeamId'])] = elo[g['hometeamId']]
        eb[(g['gameId'], g['awayteamId'])] = elo[g['awayteamId']]
        if pd.isna(g['homeScore']) or pd.isna(g['awayScore']): continue
        
        ha = elo[g['hometeamId']] + HOME_ADVANTAGE
        eh = 1.0 / (1.0 + 10 ** ((elo[g['awayteamId']] - ha) / 400))
        hw = 1.0 if g['homeScore'] > g['awayScore'] else 0.0
        
        mov = max(1, abs(g['homeScore'] - g['awayScore']))
        winner_elo = elo[g['hometeamId']] + HOME_ADVANTAGE if hw == 1.0 else elo[g['awayteamId']]
        loser_elo = elo[g['awayteamId']] if hw == 1.0 else elo[g['hometeamId']] + HOME_ADVANTAGE
        elo_diff = winner_elo - loser_elo
        
        mov_ratio = min(1.0, (mov - 1) / 29.0)
        multiplier = (1.0 + mov_ratio) * max(0.1, 1.0 - (elo_diff / 500.0))
        multiplier = min(2.0, multiplier)
        
        elo[g['hometeamId']] += K_FACTOR * multiplier * (hw - eh)
        elo[g['awayteamId']] += K_FACTOR * multiplier * ((1 - hw) - (1 - eh))"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('def compute_elo_ratings(games_df):' in line for line in source):
            full_source = ''.join(source)
            full_source = full_source.replace(old_block, new_block)
            cell['source'] = [line for line in full_source.splitlines(True)]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Updated notebook successfully.")
