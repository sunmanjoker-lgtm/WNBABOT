import requests
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from nba_api.stats.endpoints import teamgamelog

TOKEN = '8917243606:AAHojdm5VMfKCasorA05zVtVphYXyNb4n5k'
CHAT_ID = 328619258

def send_message(text):
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
    params = {'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'HTML'}
    return requests.post(url, json=params).json()

# Загружаем модели
with open('wnba_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('wnba_model_win.pkl', 'rb') as f:
    model_win = pickle.load(f)
with open('wnba_model_total.pkl', 'rb') as f:
    model_total = pickle.load(f)
with open('wnba_model_margin.pkl', 'rb') as f:
    model_margin = pickle.load(f)

wnba_teams = [
    {'abbr': 'ATL', 'id': 1611661327}, {'abbr': 'CHI', 'id': 1611661328},
    {'abbr': 'CON', 'id': 1611661329}, {'abbr': 'DAL', 'id': 1611661330},
    {'abbr': 'IND', 'id': 1611661331}, {'abbr': 'LA', 'id': 1611661332},
    {'abbr': 'LV', 'id': 1611661333}, {'abbr': 'MIN', 'id': 1611661334},
    {'abbr': 'NY', 'id': 1611661335}, {'abbr': 'PHX', 'id': 1611661336},
    {'abbr': 'SEA', 'id': 1611661337}, {'abbr': 'WAS', 'id': 1611661338},
    {'abbr': 'GSV', 'id': 1611661339}
]

def get_team_id(abbr):
    for t in wnba_teams:
        if t['abbr'] == abbr.upper():
            return t['id']
    return None

def get_last_5_avg(team_abbr):
    team_id = get_team_id(team_abbr)
    if not team_id:
        return None
    gamelog = teamgamelog.TeamGameLog(team_id=team_id, season='2026', league_id='10')
    df = gamelog.get_data_frames()[0]
    if len(df) < 5:
        return None
    recent = df.head(5)
    fga = recent['FGA'].mean()
    oreb = recent['OREB'].mean()
    tov = recent['TOV'].mean()
    fta = recent['FTA'].mean()
    pace = fga - oreb + tov + 0.4 * fta
    off_rtg = (recent['PTS'].mean() / pace) * 100 if pace > 0 else 0
    return {
        'PTS': recent['PTS'].mean(),
        'FG_PCT': recent['FG_PCT'].mean(),
        'FG3_PCT': recent['FG3_PCT'].mean(),
        'REB': recent['REB'].mean(),
        'AST': recent['AST'].mean(),
        'TOV': tov,
        'STL': recent['STL'].mean(),
        'BLK': recent['BLK'].mean(),
        'OFF_RTG': off_rtg,
        'PACE': pace
    }

def get_wnba_games_from_api(date_str):
    """date_str: 'YYYYMMDD'"""
    url = f'https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={date_str}'
    resp = requests.get(url)
    if resp.status_code != 200:
        return []
    data = resp.json()
    events = data.get('events', [])
    games = []
    for ev in events:
        comp = ev.get('competitions', [{}])[0]
        competitors = comp.get('competitors', [])
        if len(competitors) < 2:
            continue
        away = competitors[1]['team']['displayName']
        home = competitors[0]['team']['displayName']
        # Время начала
        start_date = ev.get('date')
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        start_time_str = start_dt.strftime('%H:%M')
        games.append({
            'home': home,
            'away': away,
            'time': start_time_str,
            'start_dt': start_dt
        })
    return games

# Основная логика
now = datetime.utcnow()  # ESPN возвращает UTC
today_str = now.strftime('%Y%m%d')
tomorrow_str = (now + timedelta(days=1)).strftime('%Y%m%d')

all_games = []
for d in [today_str, tomorrow_str]:
    games = get_wnba_games_from_api(d)
    all_games.extend(games)

if not all_games:
    send_message('Сегодня и завтра игр WNBA нет (по данным ESPN).')
else:
    predictions = []
    for game in all_games:
        home = game['home']
        away = game['away']
        start_dt = game['start_dt']
        hours_until = (start_dt - now).total_seconds() / 3600

        # Фильтр: от 6 до 24 часов до начала
        if hours_until < 6 or hours_until > 24:
            predictions.append(f"{home} vs {away}: пропущено (не в окне 6-24 ч.)")
            continue

        home_stats = get_last_5_avg(home)
        away_stats = get_last_5_avg(away)
        if home_stats is None or away_stats is None:
            predictions.append(f"{home} vs {away}: недостаточно данных (нет 5 игр)")
            continue

        features = {}
        for col in ['PTS','FG_PCT','FG3_PCT','REB','AST','TOV','STL','BLK','OFF_RTG','PACE']:
            features[f'DIFF_{col}'] = home_stats[col] - away_stats[col]
        X = pd.DataFrame([features])
        X_scaled = scaler.transform(X)

        prob_win = model_win.predict_proba(X_scaled)[0][1]
        pred_total = model_total.predict(X_scaled)[0]
        pred_margin = model_margin.predict(X_scaled)[0]

        win_rec = 'ставка на хозяев' if prob_win > 0.60 else ('ставка на гостей' if prob_win < 0.40 else 'пропустить')
        total_line = 160.5  # временно
        total_rec = 'ТБ' if pred_total > total_line + 2 else ('ТМ' if pred_total < total_line - 2 else 'пропустить')
        margin_rec = f'фора хозяев -{abs(pred_margin):.1f}' if pred_margin > 0 else f'фора гостей {abs(pred_margin):.1f}'

        predictions.append(
            f"{home} vs {away} (в {game['time']}):\n"
            f"  Победа: {win_rec} ({prob_win:.0%})\n"
            f"  Тотал: {pred_total:.1f} → {total_rec}\n"
            f"  Маржа: {pred_margin:.1f} → {margin_rec}"
        )

    if not predictions:
        send_message('Нет матчей, подходящих по времени или данным.')
    else:
        msg = "🏀 Прогнозы WNBA на сегодня/завтра:\n\n" + "\n\n".join(predictions)
        send_message(msg)
