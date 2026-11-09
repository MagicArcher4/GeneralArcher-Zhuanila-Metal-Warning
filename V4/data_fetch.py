# ============================================================
# 数据获取：原油用FRED，其他用AKShare
# 爬不到返回None，绝不中断
# ============================================================
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 【防崩】akshare / pandas-datareader 是重型可选依赖，
# 云端安装失败或版本冲突时不允许直接让整个网页白屏，
# 这里降级为「自动获取不可用」，手动输入仍然正常工作。
# ============================================================
try:
    import akshare as ak
except Exception as _e:
    ak = None
    print(f"⚠️ akshare 导入失败，自动获取功能不可用（可手动输入）: {_e}")

try:
    from pandas_datareader import data as pdr
except Exception as _e:
    pdr = None
    print(f"⚠️ pandas-datareader 导入失败，原油自动获取不可用（可手动输入）: {_e}")


# ============================================================
# 通用工具
# ============================================================
def _parse_chinese_month(s):
    s = str(s).strip().replace('月份', '').replace('年', '-').replace('月', '')
    parts = s.split('-')
    if len(parts) == 2:
        try:
            return f"{parts[0]}-{int(parts[1]):02d}"
        except:
            return s
    return s


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_daily_mean(df, value_col, month_str, date_col='日期'):
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        target = pd.to_datetime(month_str)
        mask = (df[date_col].dt.year == target.year) & (df[date_col].dt.month == target.month)
        sub = df[mask]
        if len(sub) == 0:
            sub = df[df[date_col] <= target].tail(1)
        if len(sub) == 0:
            return None
        v = sub[value_col].mean()
        return float(v) if not pd.isna(v) else None
    except:
        return None


def _get_monthly_value(df, value_col, month_str, date_col='月份'):
    try:
        df = df.copy()
        df['_std'] = df[date_col].apply(_parse_chinese_month)
        target = pd.to_datetime(month_str).strftime('%Y-%m')
        match = df[df['_std'] == target]
        if len(match) == 0:
            match = df.head(1)
        v = match.iloc[0][value_col]
        return float(v) if not pd.isna(v) else None
    except:
        return None


# ============================================================
# X1：原油（用FRED，最稳）
# ============================================================
def fetch_oil(month_str):
    """布伦特原油，FRED月度数据"""
    if pdr is None:
        return None
    try:
        target = pd.to_datetime(month_str)
        # 读FRED（数据只到当前月，读未来月份会返回空）
        start = (target - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        end = (target + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

        df = pdr.DataReader("MCOILBRENTEU", data_source="fred",
                            start=start, end=end)
        if df is None or len(df) == 0:
            return None

        df = df.reset_index()
        df.columns = ['date', 'oil']
        df['date'] = pd.to_datetime(df['date'])

        # 找目标月
        mask = (df['date'].dt.year == target.year) & (df['date'].dt.month == target.month)
        sub = df[mask]
        if len(sub) == 0:
            return None

        v = sub['oil'].mean()
        return float(v) if not pd.isna(v) else None
    except Exception as e:
        # 不打印，避免刷屏
        return None


# ============================================================
# X2/X4/X5：国内期货（AKShare）
# ============================================================
def fetch_copper(month_str):
    try:
        df = ak.futures_zh_daily_sina(symbol="CU0")
        if df is None or len(df) == 0:
            return None
        date_col = _find_col(df, ['date', '日期'])
        value_col = _find_col(df, ['close', '收盘价', '最新价'])
        if not date_col or not value_col:
            return None
        return _get_daily_mean(df, value_col, month_str, date_col)
    except:
        return None


def fetch_iron(month_str):
    try:
        df = ak.futures_zh_daily_sina(symbol="I0")
        if df is None or len(df) == 0:
            return None
        date_col = _find_col(df, ['date', '日期'])
        value_col = _find_col(df, ['close', '收盘价', '最新价'])
        if not date_col or not value_col:
            return None
        return _get_daily_mean(df, value_col, month_str, date_col)
    except:
        return None


def fetch_coal(month_str):
    try:
        df = ak.futures_zh_daily_sina(symbol="JM0")
        if df is None or len(df) == 0:
            return None
        date_col = _find_col(df, ['date', '日期'])
        value_col = _find_col(df, ['close', '收盘价', '最新价'])
        if not date_col or not value_col:
            return None
        return _get_daily_mean(df, value_col, month_str, date_col)
    except:
        return None


# ============================================================
# X7：工业增加值（AKShare）
# ============================================================
def fetch_industrial(month_str):
    try:
        df = ak.macro_china_gyzjz()
        if df is None or len(df) == 0:
            return None
        date_col = _find_col(df, ['月份', '日期', 'date'])
        value_col = _find_col(df, ['今值', '同比增长', '工业增加值同比增长', 'value'])
        if not date_col or not value_col:
            return None
        return _get_monthly_value(df, value_col, month_str, date_col)
    except:
        return None


# ============================================================
# X8：汇率（AKShare 中行）
# ============================================================
def fetch_fx(month_str):
    try:
        df = ak.currency_boc_sina(symbol="美元", start_date="20200101", end_date="20301231")
        if df is None or len(df) == 0:
            return None
        date_col = _find_col(df, ['日期', 'date'])
        if not date_col:
            return None
        v = _get_daily_mean(df, df.columns[-1], month_str, date_col)
        if v and v > 100:
            v = v / 100
        return v
    except:
        return None


# ============================================================
# X9：BDI（AKShare）
# ============================================================
def fetch_bdi(month_str):
    try:
        df = ak.macro_shipping_bdi()
        if df is None or len(df) == 0:
            return None
        date_col = _find_col(df, ['日期', 'date'])
        value_col = _find_col(df, ['最新值', 'BDI', 'value'])
        if not date_col or not value_col:
            return None
        return _get_daily_mean(df, value_col, month_str, date_col)
    except:
        return None


# ============================================================
# X10/X11：PMI（AKShare）
# ============================================================
def fetch_pmi(month_str):
    try:
        df = ak.macro_china_pmi()
        if df is None or len(df) == 0:
            return None, None
        pmi = _get_monthly_value(df, '制造业-指数', month_str, '月份')
        pmi_yoy = _get_monthly_value(df, '制造业-同比增长', month_str, '月份')
        return pmi, pmi_yoy
    except:
        return None, None


# ============================================================
# 总入口
# ============================================================
def auto_fetch_data(month_str):
    result = {}
    failed = []

    tasks = [
        ('X1布伦特原油价格（每桶）', fetch_oil),
        ('X2铜价格', fetch_copper),
        ('X4铁矿石价格', fetch_iron),
        ('X5焦煤价格', fetch_coal),
        ('X7工业增加值', fetch_industrial),
        ('X8 USD_CNY月度平均汇率', fetch_fx),
        ('X9 BDI', fetch_bdi),
    ]

    for key, func in tasks:
        try:
            v = func(month_str)
        except:
            v = None
        result[key] = v
        if v is None:
            failed.append(key)

    try:
        pmi, pmi_yoy = fetch_pmi(month_str)
    except:
        pmi, pmi_yoy = None, None

    result['X10 PMI（制造业-指数）'] = pmi
    result['X11 PMI（制造业-同比增长）'] = pmi_yoy
    if pmi is None:
        failed.append('X10 PMI（制造业-指数）')
    if pmi_yoy is None:
        failed.append('X11 PMI（制造业-同比增长）')

    return result, failed


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    month = '2025-12'
    print(f"测试 {month} 的自动获取...\n")
    data, failed = auto_fetch_data(month)

    print("获取结果：")
    for k, v in data.items():
        status = "✅" if v is not None else "❌"
        v_str = f"{v:.2f}" if v is not None else "None"
        print(f"  {status} {k}: {v_str}")

    if failed:
        print(f"\n失败项（需手动填）：{failed}")