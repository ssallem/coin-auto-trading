"""캔들 패턴 감지 유틸리티"""
from typing import Optional, Dict
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class CandlePatterns:

    @staticmethod
    def is_bullish_engulfing(df: pd.DataFrame, idx: int = -1) -> bool:
        """장악형 양봉: 현재봉이 양봉이고 직전 음봉의 실체를 완전히 포함"""
        try:
            if len(df) < 2:
                return False

            # iloc 인덱스로 변환
            if idx < 0:
                idx = len(df) + idx

            if idx < 1 or idx >= len(df):
                return False

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            # 현재봉이 양봉이어야 함
            current_bullish = current['close'] > current['open']
            # 직전봉이 음봉이어야 함
            prev_bearish = prev['close'] < prev['open']

            if not (current_bullish and prev_bearish):
                return False

            # 현재봉의 실체가 직전봉의 실체를 완전히 포함
            current_body_low = min(current['open'], current['close'])
            current_body_high = max(current['open'], current['close'])
            prev_body_low = min(prev['open'], prev['close'])
            prev_body_high = max(prev['open'], prev['close'])

            engulfing = (current_body_low < prev_body_low and
                        current_body_high > prev_body_high)

            if engulfing:
                logger.debug(f"Bullish engulfing detected at idx={idx}")

            return engulfing

        except Exception as e:
            logger.error(f"Error in is_bullish_engulfing: {e}")
            return False

    @staticmethod
    def is_bearish_engulfing(df: pd.DataFrame, idx: int = -1) -> bool:
        """하락 장악형: 현재봉이 음봉이고 직전 양봉의 실체를 완전히 포함"""
        try:
            if len(df) < 2:
                return False

            # iloc 인덱스로 변환
            if idx < 0:
                idx = len(df) + idx

            if idx < 1 or idx >= len(df):
                return False

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            # 현재봉이 음봉이어야 함
            current_bearish = current['close'] < current['open']
            # 직전봉이 양봉이어야 함
            prev_bullish = prev['close'] > prev['open']

            if not (current_bearish and prev_bullish):
                return False

            # 현재봉의 실체가 직전봉의 실체를 완전히 포함
            current_body_low = min(current['open'], current['close'])
            current_body_high = max(current['open'], current['close'])
            prev_body_low = min(prev['open'], prev['close'])
            prev_body_high = max(prev['open'], prev['close'])

            engulfing = (current_body_low < prev_body_low and
                        current_body_high > prev_body_high)

            if engulfing:
                logger.debug(f"Bearish engulfing detected at idx={idx}")

            return engulfing

        except Exception as e:
            logger.error(f"Error in is_bearish_engulfing: {e}")
            return False

    @staticmethod
    def count_consecutive_bearish(df: pd.DataFrame, max_lookback: int = 10) -> int:
        """최신봉부터 역순으로 연속 음봉 개수 (최신봉 제외, 그 이전부터 카운트)"""
        try:
            if len(df) < 2:
                return 0

            count = 0
            # 최신봉 제외하고 그 이전부터 카운트
            for i in range(len(df) - 2, max(len(df) - max_lookback - 2, -1), -1):
                candle = df.iloc[i]
                if candle['close'] < candle['open']:  # 음봉
                    count += 1
                else:
                    break

            if count > 0:
                logger.debug(f"Consecutive bearish candles: {count}")

            return count

        except Exception as e:
            logger.error(f"Error in count_consecutive_bearish: {e}")
            return 0

    @staticmethod
    def count_consecutive_bullish(df: pd.DataFrame, max_lookback: int = 10) -> int:
        """최신봉부터 역순으로 연속 양봉 개수 (최신봉 제외, 그 이전부터 카운트)"""
        try:
            if len(df) < 2:
                return 0

            count = 0
            # 최신봉 제외하고 그 이전부터 카운트
            for i in range(len(df) - 2, max(len(df) - max_lookback - 2, -1), -1):
                candle = df.iloc[i]
                if candle['close'] > candle['open']:  # 양봉
                    count += 1
                else:
                    break

            if count > 0:
                logger.debug(f"Consecutive bullish candles: {count}")

            return count

        except Exception as e:
            logger.error(f"Error in count_consecutive_bullish: {e}")
            return 0

    @staticmethod
    def recent_touches_lower_band(df: pd.DataFrame, lookback: int = 5) -> bool:
        """최근 lookback 봉 중 볼린저 하단밴드를 터치/이탈한 적 있는지
        터치 = low <= bb_lower"""
        try:
            if len(df) < lookback:
                return False

            if 'bb_lower' not in df.columns:
                logger.warning("bb_lower column not found in DataFrame")
                return False

            recent_df = df.iloc[-lookback:]

            # NaN 값 제외하고 체크
            valid_rows = recent_df.dropna(subset=['bb_lower', 'low'])
            if len(valid_rows) == 0:
                return False

            touches = (valid_rows['low'] <= valid_rows['bb_lower']).any()

            if touches:
                logger.debug(f"Lower band touch detected in recent {lookback} candles")

            return touches

        except Exception as e:
            logger.error(f"Error in recent_touches_lower_band: {e}")
            return False

    @staticmethod
    def recent_touches_upper_band(df: pd.DataFrame, lookback: int = 5) -> bool:
        """최근 lookback 봉 중 볼린저 상단밴드를 터치/이탈한 적 있는지
        터치 = high >= bb_upper"""
        try:
            if len(df) < lookback:
                return False

            if 'bb_upper' not in df.columns:
                logger.warning("bb_upper column not found in DataFrame")
                return False

            recent_df = df.iloc[-lookback:]

            # NaN 값 제외하고 체크
            valid_rows = recent_df.dropna(subset=['bb_upper', 'high'])
            if len(valid_rows) == 0:
                return False

            touches = (valid_rows['high'] >= valid_rows['bb_upper']).any()

            if touches:
                logger.debug(f"Upper band touch detected in recent {lookback} candles")

            return touches

        except Exception as e:
            logger.error(f"Error in recent_touches_upper_band: {e}")
            return False

    @staticmethod
    def find_double_bottom(df: pd.DataFrame, window: int = 30) -> Optional[Dict]:
        """
        쌍바닥 패턴 감지 (최근 window 봉 내).

        알고리즘:
        1. 최근 window 봉의 low 값에서 로컬 최소값 2개 찾기
           (scipy 대신 간단한 비교 로직으로 구현: 양옆보다 낮은 봉)
        2. 두 저점 가격 차이 ≤ 3% (유사한 가격대)
        3. 두 저점 사이 최소 3봉 간격
        4. 두 저점 사이에 반등 존재 (중간 고점이 저점보다 1% 이상 높음)

        볼린저 밴드 조건 (bb_lower 컬럼이 있으면):
        - 두 번째 저점의 low가 bb_lower보다 높으면 inside_band=True

        Returns:
            {
                "first_low": float,
                "second_low": float,
                "first_idx": int,      # iloc 기준 인덱스
                "second_idx": int,
                "peak_between": float, # 중간 고점
                "inside_band": bool    # 두 번째 저점이 밴드 안인지
            } 또는 None
        """
        try:
            if len(df) < window or len(df) < 7:  # 최소 7봉 필요 (order=2 양옆, 간격 3)
                return None

            # 최근 window 봉 추출
            recent_df = df.iloc[-window:].copy()

            # 로컬 최소값 찾기 (order=2: 양옆 2봉보다 낮으면 극값)
            local_mins = []
            order = 2

            for i in range(order, len(recent_df) - order):
                current_low = recent_df.iloc[i]['low']

                # 양옆 order개 봉들과 비교
                is_local_min = True
                for j in range(1, order + 1):
                    if (current_low >= recent_df.iloc[i - j]['low'] or
                        current_low >= recent_df.iloc[i + j]['low']):
                        is_local_min = False
                        break

                if is_local_min:
                    # 전체 df 기준 인덱스로 변환
                    global_idx = len(df) - window + i
                    local_mins.append({
                        'idx': global_idx,
                        'low': current_low
                    })

            if len(local_mins) < 2:
                return None

            # 가장 최근 2개의 로컬 최소값 찾기
            local_mins.sort(key=lambda x: x['idx'], reverse=True)

            # 모든 가능한 쌍 검사 (최신 것부터)
            for i in range(len(local_mins) - 1):
                second_min = local_mins[i]

                for j in range(i + 1, len(local_mins)):
                    first_min = local_mins[j]

                    # 조건 검사
                    # 1. 최소 3봉 간격
                    if second_min['idx'] - first_min['idx'] < 3:
                        continue

                    # 2. 가격 차이 ≤ 3%
                    price_diff_pct = abs(second_min['low'] - first_min['low']) / first_min['low'] * 100
                    if price_diff_pct > 3.0:
                        continue

                    # 3. 중간에 반등 존재 (중간 고점이 저점보다 1% 이상 높음)
                    between_df = df.iloc[first_min['idx']:second_min['idx'] + 1]
                    peak_between = between_df['high'].max()
                    min_low = min(first_min['low'], second_min['low'])

                    if peak_between < min_low * 1.01:  # 1% 이상 반등 없음
                        continue

                    # 볼린저 밴드 조건 체크
                    inside_band = False
                    if 'bb_lower' in df.columns:
                        second_bb_lower = df.iloc[second_min['idx']]['bb_lower']
                        if not pd.isna(second_bb_lower):
                            inside_band = second_min['low'] > second_bb_lower

                    result = {
                        'first_low': first_min['low'],
                        'second_low': second_min['low'],
                        'first_idx': first_min['idx'],
                        'second_idx': second_min['idx'],
                        'peak_between': peak_between,
                        'inside_band': inside_band
                    }

                    logger.debug(f"Double bottom detected: {result}")
                    return result

            return None

        except Exception as e:
            logger.error(f"Error in find_double_bottom: {e}")
            return None

    @staticmethod
    def find_double_top(df: pd.DataFrame, window: int = 30) -> Optional[Dict]:
        """
        쌍봉 패턴 감지 (쌍바닥의 반대).

        알고리즘:
        1. 최근 window 봉의 high 값에서 로컬 최대값 2개 찾기
        2. 두 고점 가격 차이 ≤ 3%
        3. 최소 3봉 간격
        4. 중간에 하락 존재

        볼린저 밴드 조건 (bb_upper 컬럼이 있으면):
        - 두 번째 고점의 high가 bb_upper보다 낮으면 inside_band=True

        Returns:
            유사 구조의 dict 또는 None
        """
        try:
            if len(df) < window or len(df) < 7:  # 최소 7봉 필요
                return None

            # 최근 window 봉 추출
            recent_df = df.iloc[-window:].copy()

            # 로컬 최대값 찾기 (order=2: 양옆 2봉보다 높으면 극값)
            local_maxs = []
            order = 2

            for i in range(order, len(recent_df) - order):
                current_high = recent_df.iloc[i]['high']

                # 양옆 order개 봉들과 비교
                is_local_max = True
                for j in range(1, order + 1):
                    if (current_high <= recent_df.iloc[i - j]['high'] or
                        current_high <= recent_df.iloc[i + j]['high']):
                        is_local_max = False
                        break

                if is_local_max:
                    # 전체 df 기준 인덱스로 변환
                    global_idx = len(df) - window + i
                    local_maxs.append({
                        'idx': global_idx,
                        'high': current_high
                    })

            if len(local_maxs) < 2:
                return None

            # 가장 최근 2개의 로컬 최대값 찾기
            local_maxs.sort(key=lambda x: x['idx'], reverse=True)

            # 모든 가능한 쌍 검사 (최신 것부터)
            for i in range(len(local_maxs) - 1):
                second_max = local_maxs[i]

                for j in range(i + 1, len(local_maxs)):
                    first_max = local_maxs[j]

                    # 조건 검사
                    # 1. 최소 3봉 간격
                    if second_max['idx'] - first_max['idx'] < 3:
                        continue

                    # 2. 가격 차이 ≤ 3%
                    price_diff_pct = abs(second_max['high'] - first_max['high']) / first_max['high'] * 100
                    if price_diff_pct > 3.0:
                        continue

                    # 3. 중간에 하락 존재 (중간 저점이 고점보다 1% 이상 낮음)
                    between_df = df.iloc[first_max['idx']:second_max['idx'] + 1]
                    valley_between = between_df['low'].min()
                    max_high = max(first_max['high'], second_max['high'])

                    if valley_between > max_high * 0.99:  # 1% 이상 하락 없음
                        continue

                    # 볼린저 밴드 조건 체크
                    inside_band = False
                    if 'bb_upper' in df.columns:
                        second_bb_upper = df.iloc[second_max['idx']]['bb_upper']
                        if not pd.isna(second_bb_upper):
                            inside_band = second_max['high'] < second_bb_upper

                    result = {
                        'first_high': first_max['high'],
                        'second_high': second_max['high'],
                        'first_idx': first_max['idx'],
                        'second_idx': second_max['idx'],
                        'valley_between': valley_between,
                        'inside_band': inside_band
                    }

                    logger.debug(f"Double top detected: {result}")
                    return result

            return None

        except Exception as e:
            logger.error(f"Error in find_double_top: {e}")
            return None
