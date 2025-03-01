in_pos = None
stop_loss = 0.0001
rr = 20
lots = 0.1
spread = 0.0001
strat_logs = []

for index, row in data.iterrows():
    if in_pos == 'long':
        if row.low <= actual_stop:
            # Stop loss atteint
            exit_price = actual_stop
            exit_time = index
            pnl = exit_price - entry_price
            strat_logs.append([exit_price, entry_price, in_pos, entry_time, exit_time, pnl])
            in_pos = None
        elif row.high >= take_profit:
            # Take profit atteint
            exit_price = take_profit
            exit_time = index
            pnl = exit_price - entry_price
            strat_logs.append([exit_price, entry_price, in_pos, entry_time, exit_time, pnl])
            in_pos = None
    elif in_pos == 'short':
        if row.high >= actual_stop:
            # Stop loss atteint
            exit_price = actual_stop
            exit_time = index
            pnl = entry_price - exit_price
            strat_logs.append([exit_price, entry_price, in_pos, entry_time, exit_time, pnl])
            in_pos = None
        elif row.low <= take_profit:
            # Take profit atteint
            exit_price = take_profit
            exit_time = index
            pnl = entry_price - exit_price
            strat_logs.append([exit_price, entry_price, in_pos, entry_time, exit_time, pnl])
            in_pos = None
    else:
        # Pas en position
        if row["buy_signal"]:
            # Entrée en position longue
            in_pos = 'long'
            entry_price = row.ohlc4
            actual_stop = entry_price * (1 - stop_loss)
            stop_distance = entry_price - actual_stop
            take_profit = entry_price + stop_distance * rr
            entry_time = index
        elif row["sell_signal"]:
            # Entrée en position courte
            in_pos = 'short'
            entry_price = row.ohlc4
            actual_stop = entry_price * (1 + stop_loss)
            stop_distance = actual_stop - entry_price
            take_profit = entry_price - stop_distance * rr
            entry_time = index

strat_logs = pd.DataFrame(strat_logs, columns=['exit_price', 'entry_price', 'type', 'entry_time', 'exit_time', 'pnl'])
strat_logs['value'] = (strat_logs['pnl'] - spread) * lots * 100000
strat_logs