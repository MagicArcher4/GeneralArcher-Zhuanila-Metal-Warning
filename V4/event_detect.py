# ============================================================
# 事件识别：历史序列 + 当前输入 双模式
# ============================================================
import os
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 【关键・防崩】以「本文件所在目录」为基准定位数据文件
# 避免因启动目录不同（本地 / Streamlit 云端 / 子文件夹）而报 FileNotFoundError
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================
# =========================================
# 1. 读取数据
# ============================================================
df = pd.read_excel(os.path.join(BASE_DIR, '数据库.xlsx'), sheet_name='总数据集')
df['year_month'] = pd.to_datetime(df['year_month'], format='%y-%b')
df = df.sort_values('year_month').reset_index(drop=True)

# 处理 X7（1-2月合并）
df['X7工业增加值'] = df.groupby(df['year_month'].dt.year)['X7工业增加值'].transform(
    lambda x: x.ffill().bfill()
)

price_cols = {
    'X1布伦特原油价格（每桶）': '原油',
    'X2铜价格': '铜',
    'X3铝价格': '铝',
    'X4铁矿石价格': '铁矿石',
    'X5焦煤价格': '焦煤',
    'X9 BDI': 'BDI',
}
price_cols = {k: v for k, v in price_cols.items() if k in df.columns}

print(f"数据: {len(df)} 行, {df['year_month'].min()} ~ {df['year_month'].max()}")
print(f"参与事件识别的指标: {list(price_cols.values())}")


# ============================================================
# 事件规则库（历史 + 当前 共用）——【已优化参数】
# ============================================================
def apply_event_rules(changes, pmi):
    """
    changes: dict，包含 oil/coal/iron/copper/alu/bdi/fx 的变化率
    pmi: PMI值
    返回: [(event_type, direction), ...]
    """
    events = []
    oil = changes.get('oil', 0)
    coal = changes.get('coal', 0)
    iron = changes.get('iron', 0)
    copper = changes.get('copper', 0)
    alu = changes.get('alu', 0)
    bdi = changes.get('bdi', 0)
    fx = changes.get('fx', 0)

    # 规则1：能源供给冲击（放宽为 or，阈值降至0.02/0.01）
    if oil > 0.02 or coal > 0.02 or iron > 0.01:
        events.append(('能源供给冲击', '成本上升'))

    # 规则2：需求拉动（放宽为 or，阈值降至0.02）
    if copper > 0.02 or alu > 0.02 or pmi > 50.5:
        events.append(('需求拉动', '成本上升'))

    # 规则3：输入性通胀（阈值降低）
    if oil > 0.02 and fx > 0.01:
        events.append(('输入性通胀', '成本上升'))

    # 规则4：需求萎缩（放宽为 or，阈值降低）
    if oil < -0.02 or coal < -0.02 or iron < -0.01:
        events.append(('需求萎缩', '成本下降'))

    # 规则5：全球贸易活跃（阈值降低）
    if bdi > 0.05 and oil > 0.02:
        events.append(('全球贸易活跃', '成本上升'))

    # 规则6：单品供给冲击（原油）（放宽绝对值，3%波动就算）
    if abs(oil) > 0.03 and abs(coal) < 0.02 and abs(copper) < 0.02:
        events.append(('单品供给冲击（原油）', '成本上升' if oil > 0 else '成本下降'))

    return events


def build_event_json(date_str, event_type, direction, changes, source='historical'):
    """把事件包装成标准JSON"""
    name_map = {
        'oil': 'X1布伦特原油', 'coal': 'X5焦煤', 'iron': 'X4铁矿石',
        'copper': 'X2铜', 'alu': 'X3铝', 'bdi': 'X9 BDI', 'fx': 'X8 汇率'
    }
    affected = []
    for key, name in name_map.items():
        if abs(changes.get(key, 0)) > 0.03:
            affected.append({
                'name': name,
                'change': f"{changes[key] * 100:+.1f}%"
            })

    magnitude = '高' if abs(changes.get('oil', 0)) > 0.1 else (
        '中' if abs(changes.get('oil', 0)) > 0.05 else '低')

    return {
        # ⚠️【核心修改】加入了 event_type 以保证 event_id 唯一
        'event_id': f"EVT_{date_str.replace('-', '_')}_{event_type}_{source}",
        'time': date_str,
        'event_type': event_type,
        'direction': direction,
        'magnitude': magnitude,
        'affected_indicators': affected,
        'inference_confidence': 0.7,
        'evidence': f"多指标在{date_str}同步异动",
        'source_type': source
    }


# ============================================================
# 模式A：历史序列事件识别
# ============================================================
def detect_changepoints(series, window=6, threshold=0.1):
    values = series.dropna().values
    cps = []
    for i in range(window, len(values) - window):
        before = values[i - window:i].mean()
        after = values[i:i + window].mean()
        if before != 0:
            change = abs(after - before) / abs(before)
            if change > threshold:
                if not cps or i - cps[-1] > window:
                    cps.append(i)
    return cps


def classify_changepoint(series, cp_index, window=6):
    series = series.values
    before = series[max(0, cp_index - window):cp_index]
    after = series[cp_index:min(len(series), cp_index + window)]
    if len(before) < 2 or len(after) < 2:
        return '未知', 0
    before_mean = before.mean()
    after_mean = after.mean()
    change_pct = (after_mean - before_mean) / before_mean if before_mean != 0 else 0
    if abs(change_pct) < 0.05:
        return '正常波动', change_pct
    elif abs(change_pct) < 0.15:
        return '小幅突变', change_pct
    else:
        if abs(after[-1] - before_mean) / before_mean < 0.05:
            return '脉冲型', change_pct
        else:
            return '阶跃型', change_pct


def detect_historical_events(df, price_cols):
    """跑历史事件识别，返回事件列表"""
    print("\n" + "=" * 60)
    print("模式A：历史序列事件识别")
    print("=" * 60)

    changepoints = {}
    for col, name in price_cols.items():
        cp = detect_changepoints(df[col], window=6, threshold=0.12)
        changepoints[col] = cp

    event_candidates = []
    for col, name in price_cols.items():
        for cp in changepoints[col]:
            if cp >= len(df):
                continue
            form, magnitude = classify_changepoint(df[col], cp)
            if form in ['脉冲型', '阶跃型', '小幅突变']:
                event_candidates.append({
                    'date': df['year_month'].iloc[cp],
                    'indicator': name,
                    'col': col,
                    'index': cp
                })
    print(f"共识别出 {len(event_candidates)} 个候选突变点")

    structured_events = []
    seen_dates = set()
    for cand in event_candidates:
        idx = cand['index']
        date_str = cand['date'].strftime('%Y-%m')
        if date_str in seen_dates:
            continue

        def pct_change(col, idx, lag=1):
            if idx < lag:
                return 0
            v_now = df[col].iloc[idx]
            v_prev = df[col].iloc[idx - lag]
            return (v_now - v_prev) / v_prev if v_prev != 0 else 0

        changes = {
            'oil': pct_change('X1布伦特原油价格（每桶）', idx) if 'X1布伦特原油价格（每桶）' in df.columns else 0,
            'coal': pct_change('X5焦煤价格', idx) if 'X5焦煤价格' in df.columns else 0,
            'iron': pct_change('X4铁矿石价格', idx) if 'X4铁矿石价格' in df.columns else 0,
            'copper': pct_change('X2铜价格', idx) if 'X2铜价格' in df.columns else 0,
            'alu': pct_change('X3铝价格', idx) if 'X3铝价格' in df.columns else 0,
            'bdi': pct_change('X9 BDI', idx) if 'X9 BDI' in df.columns else 0,
            'fx': pct_change('X8 USD_CNY月度平均汇率', idx) if 'X8 USD_CNY月度平均汇率' in df.columns else 0,
        }
        pmi = df['X10 PMI（制造业-指数）'].iloc[idx] if 'X10 PMI（制造业-指数）' in df.columns else 50

        events = apply_event_rules(changes, pmi)
        if events:
            seen_dates.add(date_str)
            for event_type, direction in events:
                structured_events.append(
                    build_event_json(date_str, event_type, direction, changes, 'historical')
                )

    print(f"历史事件库：{len(structured_events)} 条事件")
    return structured_events


# ============================================================
# 模式B：当前输入事件识别（多窗口对比）
# ============================================================
def detect_current_events(X_dict, current_month, df_history):
    current = pd.to_datetime(current_month)
    history = df_history[df_history['year_month'] < current].copy()

    if len(history) < 12:
        print(f"⚠️ 历史数据不足12个月，无法对比")
        return [], {}

    col_to_key = {
        'X1布伦特原油价格（每桶）': 'oil',
        'X5焦煤价格': 'coal',
        'X4铁矿石价格': 'iron',
        'X2铜价格': 'copper',
        'X3铝价格': 'alu',
        'X9 BDI': 'bdi',
        'X8 USD_CNY月度平均汇率': 'fx',
    }

    changes = {}
    details = {}

    for col, key in col_to_key.items():
        if col not in X_dict or col not in history.columns:
            changes[key] = 0
            continue

        current_val = X_dict[col]
        mean_3m = history.tail(3)[col].mean()
        change_3m = (current_val - mean_3m) / mean_3m if mean_3m != 0 else 0

        mean_12m = history.tail(12)[col].mean()
        change_12m = (current_val - mean_12m) / mean_12m if mean_12m != 0 else 0

        pct_rank = (history[col] < current_val).mean()
        extreme_pct = 0
        if pct_rank > 0.9:
            extreme_pct = (current_val - history[col].quantile(0.9)) / history[col].quantile(0.9)
        elif pct_rank < 0.1:
            extreme_pct = (current_val - history[col].quantile(0.1)) / history[col].quantile(0.1)

        candidates = [change_3m, change_12m, extreme_pct]
        changes[key] = max(candidates, key=abs)

        details[key] = {
            'current': round(current_val, 2),
            'vs_3m': f"{change_3m * 100:+.1f}%",
            'vs_12m': f"{change_12m * 100:+.1f}%",
            'historical_pct': f"{pct_rank * 100:.1f}%"
        }

    pmi = X_dict.get('X10 PMI（制造业-指数）', 50)
    events = apply_event_rules(changes, pmi)

    result = []
    for event_type, direction in events:
        result.append(
            build_event_json(current_month, event_type, direction, changes, 'current')
        )

    return result, details


# ============================================================
# 主流程（仅测试用，网页端不会执行这里）
# ============================================================
if __name__ == '__main__':
    hist_events = detect_historical_events(df, price_cols)
    event_db = pd.DataFrame([{
        'event_id': e['event_id'],
        'time': e['time'],
        'event_type': e['event_type'],
        'direction': e['direction'],
        'magnitude': e['magnitude'],
        'evidence': e['evidence'],
        'source_type': e['source_type']
    } for e in hist_events])
    event_db.to_csv(os.path.join(BASE_DIR, 'event_db.csv'), index=False, encoding='utf-8-sig')
    print(f"\n历史事件库已保存到 event_db.csv（{len(event_db)} 条）")
