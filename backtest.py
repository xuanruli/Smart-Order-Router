import pandas as pd
import numpy as np
import json

def l1_reader(file_path):
    
    df = pd.read_csv(file_path, usecols=['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00'])

    df.sort_values(by=['ts_event', 'publisher_id'], inplace=True)

    df.drop_duplicates(inplace=True, subset=['ts_event', 'publisher_id'])
        
    return df

def mock_venue_data(row, n_venues, price_quantile, fee_step, rebate_step):
    venues = []
    for i in range(n_venues):
        idx = np.searchsorted(price_quantile, row['ask_px_00'])
        idx_shift = np.random.randint(-10, 10)
        new_idx = np.clip(idx + idx_shift, 0, len(price_quantile) - 1)
        
        venue = {
            'ask_px': round(price_quantile[new_idx]* np.random.uniform(0.99, 1.01), 2),
            'ask_sz': max(1, int(row['ask_sz_00'] * np.random.uniform(0.8, 1.5))),
            'fee': fee_step[i],
            'rebate': rebate_step[i],
        }
        venues.append(venue)

    return venues

def compute_cost(split, venues, order_size, λ_over, λ_under, θ_queue):
    executed_sz = 0
    cash_spent = 0
    for i in range(len(venues)):
        exe = min(split[i], venues[i]['ask_sz'])
        executed_sz += exe
        cash_spent += exe * venues[i]['ask_px'] + venues[i]['fee'] * exe
        maker_rebate = max(split[i] - exe, 0) * venues[i]['rebate']
        cash_spent -= maker_rebate
    
    underfill = order_size - executed_sz
    overfill = max(executed_sz - order_size, 0)
    return cash_spent + λ_over * overfill + λ_under * underfill + θ_queue * (underfill + overfill)

def allocate(order_size, venues, λ_over, λ_under, θ_queue):
    step = 100
    splits = [[]]
    for i in range(len(venues)):
        new_split = []
        for alloc in splits:
            max_v = min(venues[i]['ask_sz'], order_size - sum(alloc))
            new_split.append(alloc.copy() + [max_v])
            if step >= max_v:
                new_split.append(alloc.copy() + [0])
            else:
                for s in range(0, max_v, step):
                    new_split.append(alloc.copy() + [s])
                          
        splits = new_split
    
    best_cost = float('inf')
    best_split = []

    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, λ_over, λ_under, θ_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
            
    return best_split, best_cost

def cont_kukanov(venues, remaining):
    λ_over_range = np.linspace(0.002, 0.01, 5)
    λ_under_range = np.linspace(0.002, 0.01, 5)
    θ_range = np.linspace(0.002, 0.01, 5)
    best_cost = float('inf')
    best_allocation = []
    total_liquidity = sum(venue['ask_sz'] for venue in venues)
  
    for λ_over in λ_over_range:
        for λ_under in λ_under_range:
            for θ_queue in θ_range:
                allocation, cost = allocate(min(remaining, int(total_liquidity * 0.75)), venues, λ_over, λ_under, θ_queue)
                if cost < best_cost:
                    best_cost = cost
                    best_allocation = allocation
                    best_params = (λ_over, λ_under, θ_queue)
    
    return best_allocation, best_cost, total_liquidity, best_params

def compute_twap_cost(df, order_size):
    df['ts_event_dt'] = pd.to_datetime(df['ts_event'])
    df['bucket'] = df['ts_event_dt'].dt.floor('60s')
    twap_price_bucket = df.groupby('bucket')['ask_px_00'].mean()
    twap_price = twap_price_bucket.mean()
    twap_cost = twap_price * order_size
    return twap_cost, twap_price

def compute_vwap_cost(df, order_size):
    px_array = df['ask_px_00'].values
    sz_array = df['ask_sz_00'].values
    df_avg_px = sum(px_array * sz_array) / sum(sz_array)
    vwap_cost = df_avg_px * order_size
    vwap_price = df_avg_px
    return vwap_cost, vwap_price

def compute_best_ask_cost(venues, best_ask_remaining):
    ts_best_ask_cost = sum((venue['ask_px'] + venue['fee']) * venue['ask_sz'] for venue in venues)
    total_liquidity = sum(venue['ask_sz'] for venue in venues)
    if best_ask_remaining >= total_liquidity:
        return ts_best_ask_cost, total_liquidity
    else:
        sorted_venues = sorted(venues, key=lambda v: v['ask_px'])
        remaining = best_ask_remaining
        sub_cost, sub_quantity = 0, 0
    
        for venue in sorted_venues:
            take = min(venue['ask_sz'], remaining)
            sub_cost += (venue['ask_px'] + venue['fee']) * take
            sub_quantity += take
            remaining -= take
            if remaining <= 0:
                break
        
        return sub_cost, sub_quantity

if __name__ == '__main__':
    df = l1_reader('l1_day.csv')

    n_venues = 10
    rebate_range, fee_range = [0.002, 0.0008], [0.003, 0.0015]
    price_quantile = np.quantile(df['ask_px_00'], np.linspace(0, 1, 500))
    rebate_step = np.linspace(rebate_range[0], rebate_range[1], n_venues)
    fee_step = np.linspace(fee_range[0], fee_range[1], n_venues)

    # execution router
    order_size = 5000
    remaining, best_ask_remaining = order_size, order_size
    total_cost, count, best_ask_cost, min_cost= 0, 0, 0, float('inf')
    best_params = None
    order_counts, cumulative_costs, best_ask_costs = [], [], []
    iteration_count = []

    for _, row in df.iterrows():
        if remaining <= 0:
            break

        venues = mock_venue_data(row, n_venues, price_quantile, fee_step, rebate_step)

        # Cont & Kukanov
        best_allocation, best_cost, total_liquidity, params = cont_kukanov(venues, remaining)
        if best_cost < min_cost:
            best_params = params
            
        remaining -=sum(best_allocation)
        total_cost += best_cost
        count += 1
    
        ts_best_ask_cost, best_ask_quantity = compute_best_ask_cost(venues, best_ask_remaining)
        best_ask_remaining -= best_ask_quantity
        if best_ask_remaining >= 0:
            best_ask_cost += ts_best_ask_cost

    # baseline
    twap_cost, twap_price = compute_twap_cost(df, order_size)
    vwap_cost, vwap_price = compute_vwap_cost(df, order_size)

    # saving
    router_avg_price = total_cost / order_size
    best_ask_price = best_ask_cost / order_size

    result = {
        'best_params': {
            'overfill_param': best_params[0],
            'underfill_param': best_params[1],
            'queue_param': best_params[2]
        },
        'router_porformance': {
            'total_cost': round(total_cost, 2),
            'ts_count': count,
            'remaining': remaining
        },
        'baseline_performance': {
            'best_ask_cost': round(best_ask_cost, 2),
            'vwap_cost': round(vwap_cost, 2),
            'twap_cost': round(twap_cost, 2)
        },
        'savings': {
            'best_ask_savings': round(((best_ask_price - router_avg_price) / (best_ask_price)) * 10000, 2),
            'vwap_savings': round(((vwap_price - router_avg_price) / (vwap_price)) * 10000, 2),
            'twap_savings': round((twap_price - router_avg_price) / (twap_price) * 10000, 2)
        }
    }

    print(json.dumps(result, indent=3))


