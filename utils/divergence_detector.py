"""다이버전스 감지 + MACD 크로스 유틸리티"""
from typing import Optional, Dict
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class DivergenceDetector:
    """
    다이버전스 및 MACD 크로스 패턴 감지 클래스

    다이버전스: 가격과 지표(RSI) 사이의 방향성 불일치를 감지하여
    추세 전환 신호를 포착한다.
    """

    @staticmethod
    def _find_local_minima(
        series: pd.Series,
        order: int = 2
    ) -> np.ndarray:
        """
        시리즈에서 로컬 최소값의 인덱스를 찾는다.

        로컬 최소값 = 양쪽 order개 값보다 작은 값

        Args:
            series: 검색할 시리즈
            order: 비교할 양쪽 데이터 개수

        Returns:
            로컬 최소값의 iloc 인덱스 배열
        """
        minima_indices = []

        for i in range(order, len(series) - order):
            val = series.iloc[i]

            # NaN 체크
            if pd.isna(val):
                continue

            # 양쪽 order개 값들과 비교
            is_minimum = True
            for j in range(1, order + 1):
                left_val = series.iloc[i - j]
                right_val = series.iloc[i + j]

                # NaN이 있으면 스킵
                if pd.isna(left_val) or pd.isna(right_val):
                    is_minimum = False
                    break

                # 양쪽보다 작아야 함
                if val >= left_val or val >= right_val:
                    is_minimum = False
                    break

            if is_minimum:
                minima_indices.append(i)

        return np.array(minima_indices)

    @staticmethod
    def _find_local_maxima(
        series: pd.Series,
        order: int = 2
    ) -> np.ndarray:
        """
        시리즈에서 로컬 최대값의 인덱스를 찾는다.

        로컬 최대값 = 양쪽 order개 값보다 큰 값

        Args:
            series: 검색할 시리즈
            order: 비교할 양쪽 데이터 개수

        Returns:
            로컬 최대값의 iloc 인덱스 배열
        """
        maxima_indices = []

        for i in range(order, len(series) - order):
            val = series.iloc[i]

            # NaN 체크
            if pd.isna(val):
                continue

            # 양쪽 order개 값들과 비교
            is_maximum = True
            for j in range(1, order + 1):
                left_val = series.iloc[i - j]
                right_val = series.iloc[i + j]

                # NaN이 있으면 스킵
                if pd.isna(left_val) or pd.isna(right_val):
                    is_maximum = False
                    break

                # 양쪽보다 커야 함
                if val <= left_val or val <= right_val:
                    is_maximum = False
                    break

            if is_maximum:
                maxima_indices.append(i)

        return np.array(maxima_indices)

    @staticmethod
    def detect_bullish_divergence(
        df: pd.DataFrame,
        window: int = 50,
        min_distance: int = 5
    ) -> Optional[Dict]:
        """
        상승 다이버전스 감지: 가격 저점은 내려가는데 RSI 저점은 올라감

        알고리즘:
        1. 최근 window 봉에서 가격(low)의 로컬 최소값들을 찾기
           (로컬 최소 = 양쪽 2봉보다 작은 값)
        2. 같은 위치의 RSI 값 확인
        3. 가장 최근 2개의 저점 쌍을 비교:
           - 가격: 두 번째 저점 < 첫 번째 저점 (하락 추세)
           - RSI: 두 번째 저점 > 첫 번째 저점 (상승 추세)
        4. 두 저점 간 거리 >= min_distance

        Args:
            df: OHLCV + RSI 데이터프레임
            window: 분석할 최근 봉 개수
            min_distance: 두 저점 사이의 최소 봉 간격

        Returns:
            {
                "price_low1": float, "price_low2": float,
                "rsi_low1": float, "rsi_low2": float,
                "idx1": int, "idx2": int,  # iloc 인덱스
                "strength": float  # 다이버전스 강도 0~1
            } 또는 None
        """
        # 입력 검증
        if len(df) < window + 4:  # order=2이므로 양쪽 2개씩 필요
            logger.debug(f"DataFrame 길이 부족: {len(df)} < {window + 4}")
            return None

        if 'low' not in df.columns or 'rsi' not in df.columns:
            logger.warning("필수 컬럼 누락: 'low' 또는 'rsi'")
            return None

        # 최근 window 봉 추출
        recent_df = df.iloc[-window:].copy()

        # low 시리즈에서 로컬 최소값 찾기
        low_series = recent_df['low']
        rsi_series = recent_df['rsi']

        minima_indices = DivergenceDetector._find_local_minima(low_series, order=2)

        if len(minima_indices) < 2:
            logger.debug(f"로컬 최소값 부족: {len(minima_indices)}개 발견")
            return None

        logger.debug(f"로컬 최소값 {len(minima_indices)}개 발견: {minima_indices}")

        # 최근 2개의 저점을 선택 (역순으로 정렬)
        minima_indices = sorted(minima_indices)

        # 가장 최근 2개를 찾되, min_distance 조건 만족하는 쌍 찾기
        for i in range(len(minima_indices) - 1, 0, -1):
            idx2 = minima_indices[i]  # 더 최근 저점

            for j in range(i - 1, -1, -1):
                idx1 = minima_indices[j]  # 더 과거 저점

                # 거리 조건 확인
                if idx2 - idx1 < min_distance:
                    continue

                # 가격과 RSI 값 추출
                price_low1 = low_series.iloc[idx1]
                price_low2 = low_series.iloc[idx2]
                rsi_low1 = rsi_series.iloc[idx1]
                rsi_low2 = rsi_series.iloc[idx2]

                # NaN 체크
                if pd.isna(rsi_low1) or pd.isna(rsi_low2):
                    logger.debug(f"RSI NaN 발견: idx1={idx1}, idx2={idx2}")
                    continue

                # 다이버전스 조건 확인
                # 가격: 두 번째 저점 < 첫 번째 저점 (하락)
                # RSI: 두 번째 저점 > 첫 번째 저점 (상승)
                if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                    # 다이버전스 강도 계산
                    # RSI 상승률 / 가격 하락률의 비율
                    price_change_pct = abs((price_low2 - price_low1) / price_low1) * 100
                    rsi_change = rsi_low2 - rsi_low1

                    # 강도: RSI 변화량을 가격 변화율로 정규화 (0~1 범위로 제한)
                    strength = min(1.0, rsi_change / max(price_change_pct, 0.1))

                    # 전체 DataFrame 기준 인덱스로 변환
                    global_idx1 = len(df) - window + idx1
                    global_idx2 = len(df) - window + idx2

                    result = {
                        "price_low1": float(price_low1),
                        "price_low2": float(price_low2),
                        "rsi_low1": float(rsi_low1),
                        "rsi_low2": float(rsi_low2),
                        "idx1": global_idx1,
                        "idx2": global_idx2,
                        "strength": float(strength)
                    }

                    logger.debug(
                        f"상승 다이버전스 감지: "
                        f"가격 {price_low1:.2f} → {price_low2:.2f} "
                        f"RSI {rsi_low1:.2f} → {rsi_low2:.2f} "
                        f"강도={strength:.3f}"
                    )

                    return result

        logger.debug("상승 다이버전스 조건 불만족")
        return None

    @staticmethod
    def detect_bearish_divergence(
        df: pd.DataFrame,
        window: int = 50,
        min_distance: int = 5
    ) -> Optional[Dict]:
        """
        하락 다이버전스 감지: 가격 고점은 올라가는데 RSI 고점은 내려감

        알고리즘:
        1. 최근 window 봉에서 가격(high)의 로컬 최대값들을 찾기
        2. 같은 위치의 RSI 값 확인
        3. 가장 최근 2개의 고점 쌍을 비교:
           - 가격: 두 번째 고점 > 첫 번째 고점 (상승 추세)
           - RSI: 두 번째 고점 < 첫 번째 고점 (하락 추세)
        4. 간격 >= min_distance

        Args:
            df: OHLCV + RSI 데이터프레임
            window: 분석할 최근 봉 개수
            min_distance: 두 고점 사이의 최소 봉 간격

        Returns:
            {
                "price_high1": float, "price_high2": float,
                "rsi_high1": float, "rsi_high2": float,
                "idx1": int, "idx2": int,  # iloc 인덱스
                "strength": float  # 다이버전스 강도 0~1
            } 또는 None
        """
        # 입력 검증
        if len(df) < window + 4:
            logger.debug(f"DataFrame 길이 부족: {len(df)} < {window + 4}")
            return None

        if 'high' not in df.columns or 'rsi' not in df.columns:
            logger.warning("필수 컬럼 누락: 'high' 또는 'rsi'")
            return None

        # 최근 window 봉 추출
        recent_df = df.iloc[-window:].copy()

        # high 시리즈에서 로컬 최대값 찾기
        high_series = recent_df['high']
        rsi_series = recent_df['rsi']

        maxima_indices = DivergenceDetector._find_local_maxima(high_series, order=2)

        if len(maxima_indices) < 2:
            logger.debug(f"로컬 최대값 부족: {len(maxima_indices)}개 발견")
            return None

        logger.debug(f"로컬 최대값 {len(maxima_indices)}개 발견: {maxima_indices}")

        # 최근 2개의 고점을 선택
        maxima_indices = sorted(maxima_indices)

        # 가장 최근 2개를 찾되, min_distance 조건 만족하는 쌍 찾기
        for i in range(len(maxima_indices) - 1, 0, -1):
            idx2 = maxima_indices[i]  # 더 최근 고점

            for j in range(i - 1, -1, -1):
                idx1 = maxima_indices[j]  # 더 과거 고점

                # 거리 조건 확인
                if idx2 - idx1 < min_distance:
                    continue

                # 가격과 RSI 값 추출
                price_high1 = high_series.iloc[idx1]
                price_high2 = high_series.iloc[idx2]
                rsi_high1 = rsi_series.iloc[idx1]
                rsi_high2 = rsi_series.iloc[idx2]

                # NaN 체크
                if pd.isna(rsi_high1) or pd.isna(rsi_high2):
                    logger.debug(f"RSI NaN 발견: idx1={idx1}, idx2={idx2}")
                    continue

                # 다이버전스 조건 확인
                # 가격: 두 번째 고점 > 첫 번째 고점 (상승)
                # RSI: 두 번째 고점 < 첫 번째 고점 (하락)
                if price_high2 > price_high1 and rsi_high2 < rsi_high1:
                    # 다이버전스 강도 계산
                    # RSI 하락량 / 가격 상승률의 비율
                    price_change_pct = abs((price_high2 - price_high1) / price_high1) * 100
                    rsi_change = abs(rsi_high2 - rsi_high1)

                    # 강도: RSI 변화량을 가격 변화율로 정규화 (0~1 범위로 제한)
                    strength = min(1.0, rsi_change / max(price_change_pct, 0.1))

                    # 전체 DataFrame 기준 인덱스로 변환
                    global_idx1 = len(df) - window + idx1
                    global_idx2 = len(df) - window + idx2

                    result = {
                        "price_high1": float(price_high1),
                        "price_high2": float(price_high2),
                        "rsi_high1": float(rsi_high1),
                        "rsi_high2": float(rsi_high2),
                        "idx1": global_idx1,
                        "idx2": global_idx2,
                        "strength": float(strength)
                    }

                    logger.debug(
                        f"하락 다이버전스 감지: "
                        f"가격 {price_high1:.2f} → {price_high2:.2f} "
                        f"RSI {rsi_high1:.2f} → {rsi_high2:.2f} "
                        f"강도={strength:.3f}"
                    )

                    return result

        logger.debug("하락 다이버전스 조건 불만족")
        return None

    @staticmethod
    def is_macd_golden_cross(
        df: pd.DataFrame,
        lookback: int = 3
    ) -> bool:
        """
        MACD 골든 크로스: 최근 lookback 봉 내에서 MACD 선이 시그널 선을 상향 돌파

        조건:
        1. 현재: macd > macd_signal
        2. lookback 봉 이내에 macd <= macd_signal인 봉이 존재

        Args:
            df: macd, macd_signal 컬럼이 있는 DataFrame
            lookback: 크로스 확인 범위 (최근 n개 봉)

        Returns:
            골든 크로스 발생 여부
        """
        # 입력 검증
        if len(df) < lookback + 1:
            logger.debug(f"DataFrame 길이 부족: {len(df)} < {lookback + 1}")
            return False

        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            logger.warning("필수 컬럼 누락: 'macd' 또는 'macd_signal'")
            return False

        # 현재 값 (가장 최근 봉)
        current_macd = df['macd'].iloc[-1]
        current_signal = df['macd_signal'].iloc[-1]

        # NaN 체크
        if pd.isna(current_macd) or pd.isna(current_signal):
            logger.debug("현재 MACD 또는 Signal 값이 NaN")
            return False

        # 조건 1: 현재 MACD > Signal
        if current_macd <= current_signal:
            return False

        # 조건 2: lookback 범위 내에 MACD <= Signal인 봉 존재
        lookback_range = df.iloc[-(lookback + 1):-1]  # 현재 봉 제외

        for idx in range(len(lookback_range)):
            macd_val = lookback_range['macd'].iloc[idx]
            signal_val = lookback_range['macd_signal'].iloc[idx]

            # NaN 체크
            if pd.isna(macd_val) or pd.isna(signal_val):
                continue

            # MACD가 Signal 이하인 봉 발견
            if macd_val <= signal_val:
                logger.debug(
                    f"MACD 골든 크로스 감지: "
                    f"{lookback} 봉 전에 교차점 발견 "
                    f"(현재 MACD={current_macd:.4f} > Signal={current_signal:.4f})"
                )
                return True

        logger.debug("MACD 골든 크로스 조건 불만족: lookback 범위 내 교차점 없음")
        return False

    @staticmethod
    def is_macd_death_cross(
        df: pd.DataFrame,
        lookback: int = 3
    ) -> bool:
        """
        MACD 데드 크로스: 최근 lookback 봉 내에서 MACD 선이 시그널 선을 하향 돌파

        조건:
        1. 현재: macd < macd_signal
        2. lookback 봉 이내에 macd >= macd_signal인 봉이 존재

        Args:
            df: macd, macd_signal 컬럼이 있는 DataFrame
            lookback: 크로스 확인 범위 (최근 n개 봉)

        Returns:
            데드 크로스 발생 여부
        """
        # 입력 검증
        if len(df) < lookback + 1:
            logger.debug(f"DataFrame 길이 부족: {len(df)} < {lookback + 1}")
            return False

        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            logger.warning("필수 컬럼 누락: 'macd' 또는 'macd_signal'")
            return False

        # 현재 값 (가장 최근 봉)
        current_macd = df['macd'].iloc[-1]
        current_signal = df['macd_signal'].iloc[-1]

        # NaN 체크
        if pd.isna(current_macd) or pd.isna(current_signal):
            logger.debug("현재 MACD 또는 Signal 값이 NaN")
            return False

        # 조건 1: 현재 MACD < Signal
        if current_macd >= current_signal:
            return False

        # 조건 2: lookback 범위 내에 MACD >= Signal인 봉 존재
        lookback_range = df.iloc[-(lookback + 1):-1]  # 현재 봉 제외

        for idx in range(len(lookback_range)):
            macd_val = lookback_range['macd'].iloc[idx]
            signal_val = lookback_range['macd_signal'].iloc[idx]

            # NaN 체크
            if pd.isna(macd_val) or pd.isna(signal_val):
                continue

            # MACD가 Signal 이상인 봉 발견
            if macd_val >= signal_val:
                logger.debug(
                    f"MACD 데드 크로스 감지: "
                    f"{lookback} 봉 전에 교차점 발견 "
                    f"(현재 MACD={current_macd:.4f} < Signal={current_signal:.4f})"
                )
                return True

        logger.debug("MACD 데드 크로스 조건 불만족: lookback 범위 내 교차점 없음")
        return False
