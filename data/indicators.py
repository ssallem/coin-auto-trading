"""
기술적 분석 지표 계산 모듈

pandas DataFrame 기반으로 다양한 기술적 분석 지표를 계산한다.
ta 라이브러리를 활용하며, 핵심 지표의 직접 구현도 병행하여 유연성을 확보한다.

지원 지표:
  - RSI (Relative Strength Index)
  - SMA / EMA (이동평균)
  - 볼린저 밴드 (Bollinger Bands)
  - MACD (Moving Average Convergence Divergence)
  - 거래량 이동평균

입력 DataFrame 컬럼 (pyupbit 형식): open, high, low, close, volume
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands

from utils.logger import get_logger

logger = get_logger(__name__)


class Indicators:
    """
    기술적 분석 지표 계산기

    모든 메서드는 정적(static)이며, OHLCV DataFrame을 입력받아
    지표가 추가된 DataFrame 또는 지표 Series를 반환한다.

    두 가지 네이밍 컨벤션을 모두 지원한다:
      - 짧은 이름: rsi(), sma(), ema(), bollinger_bands(), macd()
      - calculate_ 접두어: calculate_rsi(), calculate_sma(), calculate_ema(), ...

    사용법:
        df = client.get_ohlcv("KRW-BTC")
        rsi = Indicators.calculate_rsi(df, period=14)
        upper, middle, lower = Indicators.calculate_bollinger_bands(df, period=20)
        df_full = Indicators.add_all_indicators(df)
    """

    # ─────────────────────────────────────
    # 입력 검증
    # ─────────────────────────────────────

    @staticmethod
    def _validate_df(df: pd.DataFrame, column: str = "close") -> bool:
        """
        입력 DataFrame의 유효성을 검증한다.

        Args:
            df: 검증할 DataFrame
            column: 필수 컬럼명

        Returns:
            유효하면 True, 아니면 False
        """
        if df is None or df.empty:
            logger.warning("지표 계산 실패: DataFrame이 비어있습니다.")
            return False
        if column not in df.columns:
            logger.warning(f"지표 계산 실패: '{column}' 컬럼이 존재하지 않습니다.")
            return False
        return True

    # ─────────────────────────────────────
    # RSI (Relative Strength Index)
    # ─────────────────────────────────────

    @staticmethod
    def calculate_rsi(
        df: pd.DataFrame,
        period: int = 14,
        column: str = "close",
    ) -> pd.Series:
        """
        RSI를 계산한다 (ta 라이브러리 사용).

        RSI = 100 - (100 / (1 + RS))
        RS = 평균 상승폭 / 평균 하락폭

        Args:
            df: OHLCV DataFrame
            period: RSI 계산 기간 (기본 14)
            column: 기준 컬럼명 (기본 "close")

        Returns:
            RSI 값 Series (0~100). 입력이 유효하지 않으면 빈 Series
        """
        if not Indicators._validate_df(df, column):
            return pd.Series(dtype=float)

        try:
            indicator = RSIIndicator(close=df[column], window=period)
            rsi = indicator.rsi()
            logger.debug(f"RSI 계산 완료: period={period}, 최신값={rsi.iloc[-1]:.2f}")
            return rsi
        except Exception as e:
            logger.error(f"RSI 계산 오류: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def rsi(
        df: pd.DataFrame,
        period: int = 14,
        column: str = "close",
    ) -> pd.Series:
        """calculate_rsi의 단축 별칭."""
        return Indicators.calculate_rsi(df, period, column)

    # ─────────────────────────────────────
    # 이동평균 (SMA, EMA)
    # ─────────────────────────────────────

    @staticmethod
    def calculate_sma(
        df: pd.DataFrame,
        period: int = 20,
        column: str = "close",
    ) -> pd.Series:
        """
        단순 이동평균(SMA)을 계산한다 (ta 라이브러리 사용).

        Args:
            df: OHLCV DataFrame
            period: 이동평균 기간 (기본 20)
            column: 기준 컬럼명

        Returns:
            SMA Series. 입력이 유효하지 않으면 빈 Series
        """
        if not Indicators._validate_df(df, column):
            return pd.Series(dtype=float)

        try:
            indicator = SMAIndicator(close=df[column], window=period)
            sma = indicator.sma_indicator()
            logger.debug(f"SMA 계산 완료: period={period}")
            return sma
        except Exception as e:
            logger.error(f"SMA 계산 오류: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def sma(
        df: pd.DataFrame,
        period: int = 20,
        column: str = "close",
    ) -> pd.Series:
        """calculate_sma의 단축 별칭."""
        return Indicators.calculate_sma(df, period, column)

    @staticmethod
    def calculate_ema(
        df: pd.DataFrame,
        period: int = 20,
        column: str = "close",
    ) -> pd.Series:
        """
        지수 이동평균(EMA)을 계산한다 (ta 라이브러리 사용).

        Args:
            df: OHLCV DataFrame
            period: 이동평균 기간 (기본 20)
            column: 기준 컬럼명

        Returns:
            EMA Series. 입력이 유효하지 않으면 빈 Series
        """
        if not Indicators._validate_df(df, column):
            return pd.Series(dtype=float)

        try:
            indicator = EMAIndicator(close=df[column], window=period)
            ema = indicator.ema_indicator()
            logger.debug(f"EMA 계산 완료: period={period}")
            return ema
        except Exception as e:
            logger.error(f"EMA 계산 오류: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def ema(
        df: pd.DataFrame,
        period: int = 20,
        column: str = "close",
    ) -> pd.Series:
        """calculate_ema의 단축 별칭."""
        return Indicators.calculate_ema(df, period, column)

    @staticmethod
    def moving_average(
        df: pd.DataFrame,
        period: int = 20,
        ma_type: str = "SMA",
        column: str = "close",
    ) -> pd.Series:
        """
        이동평균을 유형에 따라 계산한다.

        Args:
            df: OHLCV DataFrame
            period: 이동평균 기간
            ma_type: "SMA" 또는 "EMA"
            column: 기준 컬럼명

        Returns:
            이동평균 Series
        """
        if ma_type.upper() == "EMA":
            return Indicators.calculate_ema(df, period, column)
        return Indicators.calculate_sma(df, period, column)

    # ─────────────────────────────────────
    # 볼린저 밴드 (Bollinger Bands)
    # ─────────────────────────────────────

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0,
        column: str = "close",
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        볼린저 밴드를 계산한다 (ta 라이브러리 사용).

        Args:
            df: OHLCV DataFrame
            period: 이동평균 기간 (기본 20)
            std: 표준편차 배수 (기본 2.0)
            column: 기준 컬럼명

        Returns:
            (upper_band, middle_band, lower_band) 튜플
            입력이 유효하지 않으면 빈 Series 3개의 튜플
        """
        empty = pd.Series(dtype=float)
        if not Indicators._validate_df(df, column):
            return empty, empty, empty

        try:
            indicator = BollingerBands(
                close=df[column],
                window=period,
                window_dev=std,
            )
            upper = indicator.bollinger_hband()
            middle = indicator.bollinger_mavg()
            lower = indicator.bollinger_lband()
            logger.debug(f"볼린저밴드 계산 완료: period={period}, std={std}")
            return upper, middle, lower
        except Exception as e:
            logger.error(f"볼린저밴드 계산 오류: {e}")
            return empty, empty, empty

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close",
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """calculate_bollinger_bands의 단축 별칭."""
        return Indicators.calculate_bollinger_bands(df, period, std_dev, column)

    # ─────────────────────────────────────
    # MACD
    # ─────────────────────────────────────

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD를 계산한다 (ta 라이브러리 사용).

        MACD Line = 단기 EMA - 장기 EMA
        Signal Line = MACD Line의 EMA
        Histogram = MACD Line - Signal Line

        Args:
            df: OHLCV DataFrame
            fast: 단기 EMA 기간 (기본 12)
            slow: 장기 EMA 기간 (기본 26)
            signal: 시그널 EMA 기간 (기본 9)
            column: 기준 컬럼명

        Returns:
            (macd_line, signal_line, histogram) 튜플
            입력이 유효하지 않으면 빈 Series 3개의 튜플
        """
        empty = pd.Series(dtype=float)
        if not Indicators._validate_df(df, column):
            return empty, empty, empty

        try:
            indicator = MACD(
                close=df[column],
                window_fast=fast,
                window_slow=slow,
                window_sign=signal,
            )
            macd_line = indicator.macd()
            signal_line = indicator.macd_signal()
            histogram = indicator.macd_diff()
            logger.debug(
                f"MACD 계산 완료: fast={fast}, slow={slow}, signal={signal}"
            )
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"MACD 계산 오류: {e}")
            return empty, empty, empty

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """calculate_macd의 단축 별칭."""
        return Indicators.calculate_macd(df, fast_period, slow_period, signal_period, column)

    # ─────────────────────────────────────
    # 거래량 지표
    # ─────────────────────────────────────

    @staticmethod
    def calculate_volume_ma(
        df: pd.DataFrame,
        period: int = 20,
    ) -> pd.Series:
        """
        거래량 단순이동평균을 계산한다.

        Args:
            df: OHLCV DataFrame
            period: 이동평균 기간 (기본 20)

        Returns:
            거래량 SMA Series. 입력이 유효하지 않으면 빈 Series
        """
        if not Indicators._validate_df(df, "volume"):
            return pd.Series(dtype=float)

        try:
            volume_ma = df["volume"].rolling(window=period, min_periods=1).mean()
            logger.debug(f"거래량 이동평균 계산 완료: period={period}")
            return volume_ma
        except Exception as e:
            logger.error(f"거래량 이동평균 계산 오류: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def volume_sma(
        df: pd.DataFrame,
        period: int = 20,
    ) -> pd.Series:
        """calculate_volume_ma의 단축 별칭."""
        return Indicators.calculate_volume_ma(df, period)

    # ─────────────────────────────────────
    # DataFrame에 지표 일괄 추가
    # ─────────────────────────────────────

    @staticmethod
    def add_all_indicators(
        df: pd.DataFrame,
        rsi_period: int = 14,
        sma_short: int = 5,
        sma_long: int = 20,
        bb_period: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        volume_ma_period: int = 20,
    ) -> pd.DataFrame:
        """
        DataFrame에 주요 기술적 지표를 일괄 추가한다.

        추가되는 컬럼:
          - rsi: RSI (Relative Strength Index)
          - sma_short, sma_long: 단기/장기 단순이동평균
          - ema_short, ema_long: 단기/장기 지수이동평균
          - bb_upper, bb_middle, bb_lower: 볼린저 밴드
          - macd, macd_signal, macd_hist: MACD
          - volume_sma: 거래량 이동평균

        Args:
            df: OHLCV DataFrame (columns: open, high, low, close, volume)
            rsi_period: RSI 기간 (기본 14)
            sma_short: 단기 이동평균 기간 (기본 5)
            sma_long: 장기 이동평균 기간 (기본 20)
            bb_period: 볼린저밴드 기간 (기본 20)
            bb_std: 볼린저밴드 표준편차 배수 (기본 2.0)
            macd_fast: MACD 단기 EMA 기간 (기본 12)
            macd_slow: MACD 장기 EMA 기간 (기본 26)
            macd_signal: MACD 시그널 기간 (기본 9)
            volume_ma_period: 거래량 이동평균 기간 (기본 20)

        Returns:
            지표가 추가된 DataFrame (원본 수정 없이 복사본 반환)
            입력이 유효하지 않으면 빈 DataFrame
        """
        if df is None or df.empty:
            logger.warning("지표 일괄 추가 실패: DataFrame이 비어있습니다.")
            return pd.DataFrame()

        # 필수 컬럼 존재 여부 확인
        required_columns = {"open", "high", "low", "close", "volume"}
        missing = required_columns - set(df.columns)
        if missing:
            logger.warning(f"지표 일괄 추가 실패: 필수 컬럼 누락 - {missing}")
            return df.copy()

        result = df.copy()

        try:
            # RSI
            result["rsi"] = Indicators.calculate_rsi(df, period=rsi_period)

            # 이동평균
            result["sma_short"] = Indicators.calculate_sma(df, period=sma_short)
            result["sma_long"] = Indicators.calculate_sma(df, period=sma_long)
            result["ema_short"] = Indicators.calculate_ema(df, period=sma_short)
            result["ema_long"] = Indicators.calculate_ema(df, period=sma_long)

            # 볼린저 밴드
            upper, middle, lower = Indicators.calculate_bollinger_bands(
                df, period=bb_period, std=bb_std
            )
            result["bb_upper"] = upper
            result["bb_middle"] = middle
            result["bb_lower"] = lower

            # MACD
            macd_line, signal_line, histogram = Indicators.calculate_macd(
                df, fast=macd_fast, slow=macd_slow, signal=macd_signal
            )
            result["macd"] = macd_line
            result["macd_signal"] = signal_line
            result["macd_hist"] = histogram

            # 거래량 이동평균
            result["volume_sma"] = Indicators.calculate_volume_ma(
                df, period=volume_ma_period
            )

            logger.info(
                f"지표 일괄 추가 완료: {len(result)}행, "
                f"추가된 컬럼={len(result.columns) - len(df.columns)}개"
            )

        except Exception as e:
            logger.error(f"지표 일괄 추가 중 오류 발생: {e}")

        return result
