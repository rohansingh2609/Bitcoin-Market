import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# Load the datasets
fear_df = pd.read_csv("fear_greed_index.csv")
trades_df = pd.read_csv("historical_data.csv")

# Preprocess dates
fear_df['date'] = pd.to_datetime(fear_df['date'])
trades_df['date'] = pd.to_datetime(trades_df['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
trades_df['date'] = pd.to_datetime(trades_df['date'])

# Merge sentiment with trades
merged = pd.merge(trades_df, fear_df[['date', 'classification']], on='date', how='left')
merged.dropna(subset=['classification'], inplace=True)

# Create additional insights
merged['Profit'] = merged['Closed PnL'] > 0
sentiment_summary = merged.groupby('classification').agg(
    avg_pnl=('Closed PnL', 'mean'),
    total_pnl=('Closed PnL', 'sum'),
    win_rate=('Profit', 'mean'),
    trade_count=('Closed PnL', 'count')
).reset_index()

# Initialize Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sentiment-Based Trader Dashboard"

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("✨ Trader Performance vs Market Sentiment", className="text-center text-primary mb-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=merged['date'].min(),
                end_date=merged['date'].max(),
                display_format='YYYY-MM-DD',
                className="mb-3"
            ),

            html.Label("Select Coin:"),
            dcc.Dropdown(
                id='coin-dropdown',
                options=[{'label': coin, 'value': coin} for coin in sorted(merged['Coin'].unique())],
                multi=True,
                placeholder="Filter by coin",
                className="mb-3"
            ),

            html.Label("Select Trade Side:"),
            dcc.Dropdown(
                id='side-dropdown',
                options=[{'label': side, 'value': side} for side in merged['Side'].unique()],
                multi=True,
                placeholder="Filter by side",
                className="mb-3"
            ),
        ], width=3),

        dbc.Col([
            dcc.Graph(id='avg-pnl-chart', className="mb-4"),
            dcc.Graph(id='total-trades-chart', className="mb-4"),
            dcc.Graph(id='pnl-timeseries'),
            dcc.Graph(id='win-rate-chart', className="mt-4")
        ], width=9)
    ]),

    dbc.Row([
        dbc.Col([
            html.H4("📋 Filtered Trade Data", className="text-secondary mt-4"),
            dash_table.DataTable(
                id='filtered-table',
                columns=[{"name": i, "id": i} for i in merged.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'fontWeight': 'bold'}
            )
        ])
    ])
], fluid=True)

# Callback
@app.callback(
    Output('avg-pnl-chart', 'figure'),
    Output('total-trades-chart', 'figure'),
    Output('pnl-timeseries', 'figure'),
    Output('win-rate-chart', 'figure'),
    Output('filtered-table', 'data'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('coin-dropdown', 'value'),
    Input('side-dropdown', 'value')
)
def update_dashboard(start_date, end_date, selected_coins, selected_sides):
    df = merged.copy()
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    if selected_coins:
        df = df[df['Coin'].isin(selected_coins)]
    if selected_sides:
        df = df[df['Side'].isin(selected_sides)]

    df['Profit'] = df['Closed PnL'] > 0

    # Average PnL chart
    avg_pnl = df.groupby('classification')['Closed PnL'].mean().reset_index()
    fig_avg = px.bar(avg_pnl, x='classification', y='Closed PnL',
                     title="Average PnL by Sentiment", color='classification')

    # Total trades chart
    total_trades = df.groupby('classification')['Closed PnL'].count().reset_index()
    fig_total = px.bar(total_trades, x='classification', y='Closed PnL',
                       title="Total Trades by Sentiment", color='classification')

    # Time series chart
    pnl_ts = df.groupby(['date', 'classification'])['Closed PnL'].sum().reset_index()
    fig_ts = px.line(pnl_ts, x='date', y='Closed PnL', color='classification',
                     title="PnL Over Time by Sentiment")

    # Win rate chart
    win_rate = df.groupby('classification')['Profit'].mean().reset_index()
    fig_win = px.bar(win_rate, x='classification', y='Profit',
                     title="Win Rate by Sentiment", color='classification',
                     labels={'Profit': 'Win Rate'})

    return fig_avg, fig_total, fig_ts, fig_win, df.to_dict('records')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
