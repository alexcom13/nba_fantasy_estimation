"""Manage and visualize data from the nba_api package

Easy way of handle data and plot graphs to show distint metrics, stats or compare nba players and teams. 

"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
import seaborn as sns
from adjustText import adjust_text
import re
from radar_plot import radar_factory

from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import PlayerGameLogs, PlayerIndex

player_index = PlayerIndex(league_id= '00', season= 2026, historical_nullable= 1).get_data_frames()[0]
id_to_position = dict(zip(player_index['PERSON_ID'], player_index['POSITION']))

baskball_ref_df = pd.read_csv('data/nba_player_stats_2026_2027.csv', index_col= 0) 
id_to_spec_position = dict(zip(baskball_ref_df['Player'], baskball_ref_df['Pos']))

rough_map = {
    'G-F': 'SF',
    'F-G': 'SF',
    'C-F': 'PF',
    'F-C': 'PF',
}

__all__ = ['basic_stat_leader']

def _add_pos_df(df: pd.DataFrame)-> pd.DataFrame:
    r"""Function to add players position to a dataframe

    Add a column with the aproximated position of every player in the dataframe. It is estimated by the nba_api package and using 
    a rouhp position estimation, that maps 'G-F' and 'F-G' to 'SF' and 'C-F' and 'F-C' to 'PF' positions.

    Parameters
    ---------
    df: pd.DataFrame
        Dataframe to add the column of column for every player. Must contain 'PLAYER_ID' and 'PLAYER_NAME' columns with the player id and player 
        name.
    
    Returns
    ------
    pd.DataFrame
        Dataframe with the column 'POSITION' added, that contains the position of every player in the paramater 'df' dataframe.
    
    """

    df_aux = df.copy()
    df_aux['POSITION'] = df_aux.apply(lambda x: id_to_position[x['PLAYER_ID']], axis = 1)
    df_aux['POSITION'] = df_aux.apply(lambda x: id_to_spec_position[x['PLAYER_NAME']] if x['PLAYER_NAME'] in id_to_spec_position.keys() else x['POSITION'], axis = 1)
    df_aux['POSITION'] = df_aux['POSITION'].apply(lambda x: rough_map[x] if re.search(r'[A-Z]-[A-Z]', x) != None else x)

    df_aux['POSITION'] = df_aux['POSITION'].astype('category')
    df_aux = df_aux[list(df.columns[:4]) + ['POSITION'] + list(df.columns[4:])]

    return df_aux    

def basic_stats_leaders(season: int = 2026, last_n_games: int = None, full_name: str = True) -> pd.DataFrame:
    r"""Returns a dataframe that contains the stats nba player leaders for the season season in the last last_n_games.

    Using a call to the nba_api package returns a dataframe with the nba players who are leader in the season, and last n games 
    of points, rebounds, asists, steals and blocks per team.
    
    Parameters
    ---------
    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders
        Default is all season.

    full_name: str
        A string parameter to indicate if the return dataframe gonna contain the the full name of the players or the nickname.

    Returns
    -------
    pd.DataFrame
        A dataframe with the nba teams as index, and every column is a statistic from the game, e.g. points, rebounds, etc., 
        with two subcolumns that contains the average stat from game and the name of the player (full name or nickname) which have report that stat. 

    
    """
    vars = ['PTS', 'REB', 'AST', 'STL', 'BLK']

    last_n_games_stats = PlayerGameLogs(season_nullable= season, 
                                        season_type_nullable= 'Regular Season', 
                                        last_n_games_nullable= last_n_games).get_data_frames()[0]

    if full_name:
        avg_stats_per_team = last_n_games_stats.groupby(['TEAM_NAME', 'PLAYER_NAME'])[vars].agg('mean').reset_index().set_index('PLAYER_NAME')
    else:
        avg_stats_per_team = last_n_games_stats.groupby(['TEAM_NAME', 'NICKNAME'])[vars].agg('mean').reset_index().set_index('NICKNAME')


    return avg_stats_per_team.groupby('TEAM_NAME').agg(['max', 'idxmax'])

def plot_basic_stats_leaders(last_n_games=None, season=2026, ax: mlt.axes.Axes = None):
    r"""Function to plot a heatmap of the stats leader per team.

    Return an Axes object with a heatmap plot of the teams and nba players leader per stats.

    Parameters
    ---------
    season: int
            The first year of season that the data gonna come frome. Defualt is current season.
            For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
            The last n games of the stats gonna count to compute the leaders. 
            Default is all season.
        
    ax: mtl.axes.Axes
        An optional axes that encapsulate all elements of an indivudual plot. If is None then
        is created in this function

    Returns
    -------
    mlt.axes.Axes
        An Axes instance with the desaire plot encapsulated.
    
    """


    vars = ['PTS', 'REB', 'AST', 'STL', 'BLK']


    stats = basic_stats_leaders(season=season, last_n_games=last_n_games, full_name=False)
    stats_text = stats.copy()
    stats_text.columns = stats_text.columns.map('_'.join)
    stats

    for var in vars:
        stats_text[var] = stats_text.apply(lambda x: x[var + '_idxmax'] + " (" + str(round(x[ var + '_max'], 2)) + ")", axis = 1)

    stats_text = stats_text[vars]

    temp_df = stats.select_dtypes(include= 'number').sort_index()
    temp_df = temp_df[['PTS', 'REB', 'AST', 'BLK', 'STL']]
    df_normalized = (temp_df - temp_df.mean()) / (temp_df.std(axis = 0))
    

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(df_normalized, annot = stats_text.sort_index(), linewidth=1, fmt="",  annot_kws={"size": 7}, cmap="coolwarm", ax = ax)

    return ax

def stats_players_games(season: int = 2026, last_n_games: int = None, per_mode: str = None) -> pd.DataFrame:
    r""" Gives back a dataframe of basic and advanced stats of all player per game.

    Does a call through the package nba_api to obtain the stats of any player who play in the period
    and the season. 
    
    Parameters
    ----------
    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    Returns 
    ------
    pd.DataFrame
        A dataframe containing the stats of any player in the season, last n games and in which mode. 
    """
    basic_vars = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'NBA_FANTASY_PTS', "FG3_PCT", 'FG3M', 'FG3A', "TOV"]
    ad_vars = ['USG_PCT', 'PACE', 'NET_RATING']


    ad_stats_players = PlayerGameLogs(season_nullable= season, 
                                      season_type_nullable= 'Regular Season', 
                                      measure_type_player_game_logs_nullable= 'Advanced', 
                                      last_n_games_nullable= last_n_games, 
                                      per_mode_simple_nullable= per_mode).get_data_frames()[0]
    
    basic_stats_players = PlayerGameLogs(season_nullable= season, 
                                         season_type_nullable= 'Regular Season', 
                                         last_n_games_nullable= last_n_games, 
                                         per_mode_simple_nullable= per_mode).get_data_frames()[0]

    return pd.merge(basic_stats_players[['PLAYER_ID', 'GAME_ID','PLAYER_NAME', 'NICKNAME', 'TEAM_NAME', 'MATCHUP', 'GAME_DATE'] + basic_vars], 
                                  ad_stats_players[['PLAYER_ID', 'GAME_ID'] + ad_vars], how = 'left', on = ['GAME_ID', 'PLAYER_ID'])

def avg_game_stats(season: int = 2026, last_n_games: int = None, per_mode: str = None)-> pd.DataFrame:
    r"""A function to get the averages of all player's stat in the interval given.

    This function agruoup by player the stats of ´stats_players_games´ and computes
    the average per stat.
    
    Parameters
    ----------
    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    
    Returns
    ------
    pd.DataFrame
        A dataframe contain players as index and their averages per stats as columns.
    """
    stats = stats_players_games(season=season, last_n_games= last_n_games, per_mode=per_mode)
    stats = _add_pos_df(stats)

    mean_cols = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV"]   
    weight_cols = ["USG_PCT", "PACE", "NET_RATING"] 

    result_df = stats.groupby('PLAYER_ID').agg(
        PLAYER_NAME = ('PLAYER_NAME', 'first'),
        NICKNAME = ('NICKNAME', 'first'),
        GP = ('PLAYER_NAME', 'size'),
        TEAM_NAME = ('TEAM_NAME', 'last'),
        POSITION = ('POSITION', 'first'),

        MIN_SUM = ("MIN", "sum"),
        FG3M_SUM = ("FG3M", "sum"),
        FG3A_SUM = ("FG3A", "sum"),
        **{f"{c}_AVG": (c, "mean") for c in mean_cols},
        **{f"{c}_WSUM": (c, "sum") for c in weight_cols},
    ).reset_index().set_index('PLAYER_NAME')

    result_df["FG3_PCT_AVG"] = result_df.FG3M_SUM / result_df.FG3A_SUM.replace(0, np.nan)
    for c in weight_cols:
        result_df[f"{c}_AVG"] = result_df[f"{c}_WSUM"] / result_df.MIN_SUM

    return result_df.drop(columns=[f"{c}_WSUM" for c in weight_cols] + ["MIN_SUM", "FG3M_SUM"])

def plot_pts_min_scatter(season: int = 2026, last_n_games: int = None, per_mode: str = None, ax: mlt.axes.Axes = None):
    r"""Plot a scatter plot of players average points and average minutes, colored by position.
    
    Using the averages stats of the function ´avg_game_stats´ creates and returns an Axes object
    with a scatter plot, of player with at leats 5 points per game and 15 minutes per game,
    of their points averages againts their average minutes per game, with their position
    distinguised by a color.

    Parameters
    ----------
    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    ax: mtl.axes.Axes
            An optional axes that encapsulate all elements of an indivudual plot. If is None then
            is created in this function
    
    Returns
    -------
    mlt.axes.Axes
        An Axes instance with the desaire plot encapsulated.

    
    """
    stats = avg_game_stats(season = season, last_n_games= last_n_games, per_mode= per_mode)
    
    temp_df = stats[(stats['PTS_AVG'] > 5) & (stats['MIN_AVG'] >= 15)].copy()
    temp_df['POSITION'] = temp_df['POSITION'].apply(lambda x: x[1] if len(x) == 2 else x) 

    temp_text = pd.concat([temp_df.nlargest(20, "PTS_AVG"), temp_df.nlargest(20, "MIN_AVG")])
    temp_text = temp_text[~temp_text.index.duplicated()]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(temp_df, x = 'MIN_AVG', y = 'PTS_AVG', hue = 'POSITION', ax = ax)

    texts =  [ax.text(r.MIN_AVG, r.PTS_AVG, name, size="small")
            for name, r in temp_text.iterrows()]

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=1), ax=ax)
    return ax

def fantasy_pts_stats(season: int = 2026, last_n_games: int = None, per_mode: int = None)-> pd.DataFrame:
    r"""Generate a dataframe with some statistics of fantasy points per player.

    Using the dataframe returned by the function ´stats_players_games´ the average,
    median and 10, 90 quantile is computed of the fantasy points grouped by player.
    
    Parameters
    ----------
    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    Returns 
    ------
    pd.DataFrame
        A dataframe containing some statistics of the fantasy points per players.

    """

    stats = stats_players_games(season = season, last_n_games= last_n_games, per_mode=per_mode)
    stats = _add_pos_df(stats)

    fantasy_pts_min_var_df = stats.groupby(['PLAYER_NAME']).agg(
        NICKNAME = ('NICKNAME', 'last'),
        POSITION = ('POSITION', 'first'),
        GAMES_PLAYED = ("PLAYER_NAME", 'size'),
        TEAM_NAME = ('TEAM_NAME', 'last'),
        MIN_AVG = ('MIN', 'mean'),
        USG_PCT_AVG = ("USG_PCT", 'mean'),
        SUM_FANTASY_PTS = ('NBA_FANTASY_PTS', 'sum'),
        FANTASY_PTS_AVG = ("NBA_FANTASY_PTS", 'mean'),
        FANTASY_PTS_STD = ("NBA_FANTASY_PTS", 'std'),
        FANTASY_PTS_P10=("NBA_FANTASY_PTS", lambda x: x.quantile(0.10)),
        FANTASY_PTS_P50=("NBA_FANTASY_PTS", "median"),
        FANTASY_PTS_P90=("NBA_FANTASY_PTS", lambda x: x.quantile(0.90))
    ).dropna()

    fantasy_pts_min_var_df["CV"] = fantasy_pts_min_var_df.FANTASY_PTS_STD / fantasy_pts_min_var_df.FANTASY_PTS_AVG
    return fantasy_pts_min_var_df

def plot_fantasy_stats(season: int = 2026, last_n_games: int = None, pero_mode: int = None, ax: mlt.axes.Axes = None)-> pd.DataFrame:
    r"""Generate a scatter plot of the average of fantasy points against the standar deviation.
    
    Using the function ´fantasy_pts_stats´, this function returns an object Axes containing 
    a scatter plot of the average fantasy points and standar deviation per game, colored
    by their position. This is helpful to see the consistency of some players and 
    if is any tendency about a good strake of antohers. 
 
    Parameters
    ----------
    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    ax: mtl.axes.Axes
        An optional axes that encapsulate all elements of an indivudual plot. If is None then
        is created in this function

    Returns
    -------
    mlt.axes.Axes
        An Axes instance with the desaire plot encapsulated.
    """
    fantasy_stats = fantasy_pts_stats(season = season, last_n_games=last_n_games, per_mode=pero_mode)

    temp_df = fantasy_stats.copy()
    temp_df = temp_df[temp_df['FANTASY_PTS_AVG'] >= 15]
    temp_df['POSITION'] = temp_df['POSITION'].apply(lambda x: x[1] if len(x) == 2 else x) 

    temp_text = pd.concat([temp_df.nlargest(20, "FANTASY_PTS_AVG"), temp_df.nlargest(20, 'FANTASY_PTS_STD'), 
                           temp_df.nsmallest(20, 'FANTASY_PTS_STD')])
    temp_text = temp_text[~temp_text.index.duplicated()]

    if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

    texts =  [ax.text(r.FANTASY_PTS_AVG, r.FANTASY_PTS_STD, name, size="small")
            for name, r in temp_text.iterrows()]

    sns.scatterplot(temp_df, x = 'FANTASY_PTS_AVG', y = 'FANTASY_PTS_STD', hue = 'POSITION', palette= 'colorblind', ax = ax)

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=1), ax=ax)

    return ax

def players_tendency(window: int =3, min_periods: int=1, prev: int=3, season: int = 2026, last_n_games: int=None, per_mode: str = None)-> pd.DataFrame:
    r"""Builds a dataframe that trace tendencies of minutes or fantasy points of all players.

    Using the data from the function ´stats_players_games´ go through the games of all players computes the dynamic mean of 
    the fantasy points recored in n, window, games with at leat min_periods and compares with the average fantasy points 
    in the last prev previous games. The same with minutes.
    
    Parameters
    ----------
    window: int
        The amount of games to compute the dynamic mean.

    min_periods: int
        The lest amount of games to start computing the dynamic mean.

    prev: int
        The amount of games to compare the dynamic mean and get the difference.

    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    
    Returns 
    -------
    pd.DataFrame
        A dataframe containing tendency of minues and fantasy points per player. 
    """
    stats = stats_players_games(season=season, last_n_games=last_n_games, per_mode=per_mode)
    stats["GAME_DATE"] = pd.to_datetime(stats["GAME_DATE"])
    stats = stats.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    g = stats.groupby("PLAYER_ID")
    stats["GAME_NUM"] = g.cumcount() + 1     
    stats["DF_MIN"] = g["MIN"].diff()

    col_r = "MIN_R" + str(window)
    stats[col_r] = g["MIN"].transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    stats["MIN_prev"] = g["MIN"].transform(lambda s: s.shift(prev).expanding().mean())
    stats["DELTA_MIN"] = stats[col_r] - stats["MIN_prev"]

    stats["DF_FANTASY_PTS"] = g["NBA_FANTASY_PTS"].diff()
    pts_col_r = "PTS_R" + str(window)
    stats[pts_col_r] = g["NBA_FANTASY_PTS"].transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    stats["FANTASY_PTS_prev"] = g["NBA_FANTASY_PTS"].transform(lambda s: s.shift(prev).expanding().mean())
    stats["DELTA_FANTASY_PTS"] = stats[pts_col_r] - stats["FANTASY_PTS_prev"]

    gp = stats.groupby("PLAYER_NAME").size().rename("GP") 

    df_result = stats.dropna(subset=["DELTA_MIN", 'DELTA_FANTASY_PTS']).groupby("PLAYER_NAME").tail(1).set_index("PLAYER_NAME")[[col_r, "MIN_prev", "DELTA_MIN", pts_col_r, 'FANTASY_PTS_prev', 'DELTA_FANTASY_PTS']]
    df_result = df_result.join(gp).sort_values(["DELTA_MIN", "MIN_prev"], ascending=False)


    return df_result

def plot_tendency_pts_min(window: int=3, min_periods: int=1, prev:int=3, season:int = 2026, last_n_games:int=None, per_mode:str = None, ax: mlt.axes.Axes = None) ->mlt.axes.Axes:
    r"""Create a scatter plot of the tendency of minutes and fantansy points the parameters and the period of time.
    
    Create a scatter plot of the difference betwen de dynamic mean and the last previous games as indicated of the 
    fantasy points and minute to determine whether or not is a change in the stats. 

    Parameters
    ----------
    window: int
        The amount of games to compute the dynamic mean.

    min_periods: int
        The lest amount of games to start computing the dynamic mean.

    prev: int
        The amount of games to compare the dynamic mean and get the difference.

    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    Returns
    -------
    mlt.axes.Axes
        An Axes instance with the desaire plot encapsulated.

    """
    tend_stats = players_tendency(window= window, min_periods= min_periods, prev= prev, season = season, last_n_games=last_n_games, per_mode=per_mode)

    tend_stats = tend_stats[(tend_stats['DELTA_MIN'] > 0) & (tend_stats['DELTA_FANTASY_PTS'] > 0)]
    temp_text = pd.concat([tend_stats.nlargest(25, 'DELTA_MIN'), tend_stats.nlargest(25, 'DELTA_FANTASY_PTS') ])
    temp_text = temp_text[~temp_text.index.duplicated()]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    texts =  [ax.text(r.DELTA_FANTASY_PTS, r.DELTA_MIN, name, size="small")
        for name, r in temp_text.iterrows()]

    sns.scatterplot(tend_stats, x = 'DELTA_FANTASY_PTS', y = 'DELTA_MIN', ax = ax)
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=1), ax=ax)

    return ax

def points_allowed_teams(season: int = 2026, last_n_games: int = None, per_mode:str = None)-> pd.DataFrame:
    r"""Compute how many fantasy points the teams allowed per poisiton and for each game.

    Returns 2 dataframes, the first is the log of games agrouped by team and position to
    get the fantasy points allowed per game, the second is a pivot table of the fantasy points
    allowed per team of every poistion in the given interval

    
     Parameters
    ----------
    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    Returns 
    ------
    tuple(pd.DataFrame, pd.DataFrame)
        A tuple of two dataframes, the first one is the fantasy points allowed per team, per position 
        and for each game of the interval. The second is a pivot table with index the teams and columns 
        the position, the values are the fantasy points allowed for each team by a specific position.
        
    """

    teams_info = pd.json_normalize(teams.get_teams())
    abb_to_id = dict(zip(teams_info['abbreviation'], teams_info['id']))
    id_to_name = dict(zip(teams_info['id'], teams_info['full_name']))


    temp_df = stats_players_games(season = season, last_n_games=last_n_games, per_mode=per_mode)

    temp_df['OPPONENT_TEAM_ABBREVIATION'] = temp_df['MATCHUP'].str.extract(r'(?:@|vs\.)\s*([A-Z]{3})')
    temp_df['OPPONENT_TEAM_ID'] = temp_df.apply(lambda x: abb_to_id[x['OPPONENT_TEAM_ABBREVIATION']], axis = 1)
    temp_df['OPPONENT_TEAM_NAME'] = temp_df.apply(lambda x: id_to_name[x['OPPONENT_TEAM_ID']], axis = 1)

    temp_df = _add_pos_df(temp_df)
    temp_df['POSITION'] = temp_df['POSITION'].apply(lambda x: x[1] if len(x) == 2 else x) 


    points_allowed_teams_df = temp_df.groupby(['OPPONENT_TEAM_ID', 'POSITION']).agg(
        TEAM_NAME = ('OPPONENT_TEAM_NAME', 'first'),
        TEAM_ABBREVIATION = ('OPPONENT_TEAM_ABBREVIATION', 'first'),
        FANTASY_POINTS_ALLOWED = ('NBA_FANTASY_PTS', 'sum'),
        FANTASY_POINTS_ALLOWED_AVG = ('NBA_FANTASY_PTS', 'mean')
    )

    pivot_points_allowed_mean = points_allowed_teams_df.reset_index().pivot(index = 'TEAM_ABBREVIATION', columns= 'POSITION', values= 'FANTASY_POINTS_ALLOWED_AVG')
    pivot_points_allowed_mean['TOTAL'] = pivot_points_allowed_mean.apply(lambda x: sum([x[pos] for pos in temp_df['POSITION'].unique()]), axis = 1)

    return points_allowed_teams_df, pivot_points_allowed_mean

def plot_points_allowed_teams(season = 2026, last_n_games: int = None, per_mode: str = None, ax: mlt.axes.Axes= None)-> mlt.axes.Axes:
    r"""Plot a heatmap of the fantasy points allowed per team by each position.

    Creat a heatmap plot of the fantasy points allowed in the given interval, the index 
    is the list of the teams and the columns each poisiton ('F', 'G', 'C'), includes a 
    total amount of fantasy points allowed.

    Parameters
    ----------
    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    ax: mtl.axes.Axes
        An optional axes that encapsulate all elements of an indivudual plot. If is None then
        is created in this function

    Returns
    -------
    mlt.axes.Axes
        An Axes instance with the desaire plot encapsulated.
    
    """
    _, pivot_pts_allowed_mean = points_allowed_teams(season = season, last_n_games=last_n_games, per_mode=per_mode)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 10))

    df_normalized = (pivot_pts_allowed_mean - pivot_pts_allowed_mean.mean()) / (pivot_pts_allowed_mean.std(axis = 0))
    df_normalized = df_normalized.sort_values(by = 'TOTAL', ascending= False)

    sns.heatmap(df_normalized,  annot= pivot_pts_allowed_mean.sort_values(by = 'TOTAL', ascending= False).values, linewidth=.5, fmt=".1f", cmap="coolwarm", ax = ax)

    return ax

def plot_player_radar(arr_players: list, season: int =2026, last_n_games: int = None, per_mode: str = None, ax: mlt.axes.Axes = None)-> mlt.axes.Axes:
    r"""Creates a radar or spider plot to see the percentiles of a player or compare two or more. 
    
    Creates a radar plot to see players percentiles in points, rebounds, asists, steals, etc., to see performance
    or compare players performance in the given interval.
    
    Parameters
    ----------
    arr_players: int
        A list of the players's percentiles to use in the plot.

    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    ax: mtl.axes.Axes
        An optional axes that encapsulate all elements of an indivudual plot. If is None then
        is created in this function

    Returns
    -------
    mlt.axes.Axes
        An Axes instance with the desaire plot encapsulated.
    """
    df = avg_game_stats(season = season, last_n_games=last_n_games, per_mode=per_mode)
    df = df[(df.MIN_AVG >= 15) & (df.GP >= 5)].copy()

    vars = ['PTS', 'REB', 'AST', 'STL', 'BLK', "FG3_PCT", 'USG_PCT', 'PACE', 'NET_RATING', 'TOV']
    lower_is_better = ['TOV']

    df_perc = pd.DataFrame({
        s: df[s + "_AVG"].rank(method="max", pct=True, ascending=(s not in lower_is_better)) * 100
        for s in vars
    })
    df_perc['PLAYER_ID'] = df['PLAYER_ID']
    # df_perc = df_perc.reset_index().set_index(['PLAYER_NAME'])

    list_ids = []

    for player in arr_players:
        list_ids.append(players.find_players_by_full_name(player)[0]['id'])

    data = df_perc[df_perc['PLAYER_ID'].isin(list_ids)].drop(columns = 'PLAYER_ID')

    theta = radar_factory(len(data.columns), frame = 'polygon')

    if ax is None:
        fig, ax = plt.subplots(figsize = (6,6), subplot_kw=dict(projection='radar'))


    colors = ['b', 'r', 'g', 'm', 'y']

    ax.set_rgrids([20, 40, 60, 80])
    for d, color in zip(data.reset_index().to_numpy(), colors):
        name = d[0]
        d = d[1:]
        ax.plot(theta, d)
        ax.fill(theta, d, facecolor=color, alpha=0.25, label = name)
    ax.set_varlabels(data.columns)

    fig.legend()

    return ax

def plot_fantasy_cum_points(season: int= 2026, last_n_games: int = None, per_mode: str = None, ax: mlt.axes.Axes = None)-> mlt.axes.Axes:
    r"""Creates a stacked bar chart of each game and the acumulative fantasy points of each of the player.

    Plot a stacked horizontal bar charrt, the index are the nba teams, and each bar contains the acumulative
    amount of fantasy points reporte of all players for each team. 
    
    Parameters
    ----------
    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    ax: mtl.axes.Axes
        An optional axes that encapsulate all elements of an indivudual plot. If is None then
        is created in this function

    Returns
    -------
    mlt.axes.Axes
        An Axes instance with the desaire plot encapsulated.
    """
    df_aux = stats_players_games(season=season, last_n_games=last_n_games, per_mode=per_mode)

    sum_points_players = df_aux.groupby(['TEAM_NAME', 'PLAYER_NAME']).agg(
        NICKNAME = ('NICKNAME', 'last'),
        GAMES_PLAYED = ('PLAYER_ID', 'size'),
        SUM_FANTASY_PTS = ('NBA_FANTASY_PTS', 'sum')
    )
    sum_points_players['FANTASY_PTS_AVG'] = sum_points_players['SUM_FANTASY_PTS'] / sum_points_players['GAMES_PLAYED']
    df = sum_points_players.reset_index().copy()
    df = df[df['FANTASY_PTS_AVG'] > 15]

    df['RANK'] = df.groupby('TEAM_NAME')['SUM_FANTASY_PTS'].rank(method='first', ascending=False).astype(int)
    order = df.groupby('TEAM_NAME')['SUM_FANTASY_PTS'].sum().sort_values().index

    n = df['RANK'].max()
    palette = sns.color_palette('pastel', n_colors=n)  

    def text_color(rgb):
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return 'white' if lum < 0.55 else 'black'

    if ax is None:
            _, ax = plt.subplots(figsize=(10, 9))

    min_width = df['SUM_FANTASY_PTS'].sum() * 0.003

    for y, team in enumerate(order):
        sub = df[df['TEAM_NAME'] == team].sort_values('RANK')
        left = 0
        for _, r in sub.iterrows():
            val = r['SUM_FANTASY_PTS']
            color = palette[r['RANK'] - 1]
            ax.barh(y = y, width = val, left=left, color=color,
                    edgecolor='white', linewidth=0.8, height=0.9)
            if val >= min_width:
                ax.text(x = left + val / 2, y = y, s = r['NICKNAME'],
                        ha='center', va='center', fontsize=7,
                        color=text_color(color), rotation=0)
            left += val

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel('SUM_FANTASY_PTS')
    plt.tight_layout()

    return ax 

def plot_interval_fantasy_leaders(stat:str = 'mean',season: int = 2026, last_n_games: mlt.axes.Axes = None, per_mode: mlt.axes.Axes= None, ax: mlt.axes.Axes = None)-> mlt.axes.Axes:
    r"""Creates a range bar plot of the fantasy points performance of the top 25 players of the given stat.

    Builds a range bar plot of the top 25 players in the given stat respect to the fantasy points, the range 
    goes from the 10th percentile to the 90th. This is helpful to see the top 25 player with best floor, average 
    or ceiling of fantasy points in a given interval.
    
    Parameters
    ----------
    stat: str
        The statistic to get the top 25 player from. Can be 'mean', 'p10' or 'p90'.

    season: int
        The first year of season that the data gonna come frome. Defualt is current season.
        For example to obtain the season '2026-2027', must be 2026

    last_n_games: int
        The last n games of the stats gonna count to compute the leaders. 
        Default is all season.

    per_mode: str
        The period of that the stats are gona be standarized. Can be 'Total', 'Per36', 'PerGame',
        'MinutesPer', 'Per100Plays', 'Per100Possessions', etc.

    ax: mtl.axes.Axes
        An optional axes that encapsulate all elements of an indivudual plot. If is None then
        is created in this function

    Returns
    -------
    mlt.axes.Axes
        An Axes instance with the desaire plot encapsulated.
    """
    df_aux = fantasy_pts_stats(season=season, last_n_games=last_n_games, per_mode=per_mode)

    stat_dict = {
        'mean': 'FANTASY_PTS_AVG',
        "p90" : 'FANTASY_PTS_P90',
        'p10': 'FANTASY_PTS_P10'
    }

    column  = stat_dict[stat]

    top = df_aux.reset_index().nlargest(25, column).sort_values(column)


    if ax is None:
            _, ax = plt.subplots(figsize=(8, 9))

    ax.hlines(top.PLAYER_NAME, top.FANTASY_PTS_P10, top.FANTASY_PTS_P90, color="gray", lw=4, alpha=0.5)
    ax.scatter(top.FANTASY_PTS_P50, top.PLAYER_NAME, color="C0", zorder=3, label="Median")
    ax.scatter(top.FANTASY_PTS_AVG, top.PLAYER_NAME, color="C1", marker="|", s=160, zorder=3, label="Average    ")
    # ax.set(xlabel="Puntos fantasy por partido (barra = P10 a P90)")
    ax.legend(frameon=True, loc = 'right')
    plt.tight_layout()

    return ax