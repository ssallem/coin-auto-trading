"""
스캘핑 전략

빈번한 매매를 통해 소폭이지만 꾸준한 수익을 추구하는 단타 전략.
OR 로직 기반으로 여러 독립적인 진입 신호 중 하나라도 충족되면 매매를 실행한다.

매수 조건 (하나만 충족해도 매수):
  1. RSI 과매도 반등: 최근 3봉 내 RSI < oversold 이후 상승 전환
  2. EMA 골든크로스: 단기 EMA가 장기 EMA를 상향 돌파
  3. 볼린저 하단 반등: 최근 2봉 내 하단 터치 후 가격 상승
  4. 거래량 급증 + 양봉: 거래량 SMA 대비 급증 + 종가 > 시가

매도 조건 (하나만 충족해도 매도):
  1. RSI 과매수 하락: 최근 3봉 내 RSI > overbought 이후 하락 전환
  2. EMA 데드크로스: 단기 EMA가 장기 EMA를 하향 돌파
  3. 볼린저 상단 이탈: 최근 2봉 내 상단 터치 후 가격 하락
  4. 거래량 급증 + 음봉: 거래량 SMA 대비 급증 + 종가 < 시가

확신도:
  - 충족 조건 1개: 0.4
  - 충족 조건 2개: 0.6
  - 충족 조건 3개: 0.8
  - 충족 조건 4개: 0.95
"""

from __future__ import annotations

from typing import List

import pandas as pd

from data.indicators import Indicators
from strategies.base_strategy import BaseStrategy, Signal, SignalResult
from utils.logger import get_logger

logger = get_logger(__name__)

# 충족 조건 수 → 확신도 매핑
_CONFIDENCE_MAP = {
    1: 0.4,
    2: 0.6,
    3: 0.8,
    4: 0.95,
}


class ScalpingStrategy(BaseStrategy):
    """
    스캘핑 매매 전략

    여러 기술적 지표를 OR 로직으로 조합하여 빈번한 매매 기회를 포착한다.
    개별 조건의 임계값을 완화하여 진입 빈도를 높이고,
    복수 조건 충족 시 확신도를 높여 포지션 크기를 조절할 수 있도록 한다.

    Args:
        rsi_period: RSI 계산 기간 (기본 7, 짧게 설정하여 빠른 신호)
        rsi_oversold: RSI 과매도 기준값 (기본 35, 완화)
        rsi_overbought: RSI 과매수 기준값 (기본 65, 완화)
        ema_fast: 단기 EMA 기간 (기본 3)
        ema_slow: 장기 EMA 기간 (기본 10)
        bb_period: 볼린저 밴드 기간 (기본 15)
        bb_std_dev: 볼린저 밴드 표준편차 (기본 2.0)
        volume_surge_ratio: 거래량 급증 판단 배수 (기본 1.5)
    """

    def __init__(
        self,
        rsi_period: int = 7,
        rsi_oversold: float = 35,
        rsi_overbought: float = 65,
        ema_fast: int = 3,
        ema_slow: int = 10,
        bb_period: int = 15,
        bb_std_dev: float = 2.0,
        volume_surge_ratio: float = 1.5,
    ) -> None:
        self._rsi_period = rsi_period
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow
        self._bb_period = bb_period
        self._bb_std_dev = bb_std_dev
        self._volume_surge_ratio = volume_surge_ratio

        logger.info(
            "스캘핑 전략 초기화: rsi_period=%d, rsi(%s/%s), "
            "ema(%d/%d), bb(%d/%.1f), vol_ratio=%.1f",
            self._rsi_period,
            self._rsi_oversold,
            self._rsi_overbought,
            self._ema_fast,
            self._ema_slow,
            self._bb_period,
            self._bb_std_dev,
            self._volume_surge_ratio,
        )

    # ------------------------------------------------------------------ #
    #  BaseStrategy 인터페이스
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "scalping"

    def analyze(
        self,
        market: str,
        df: pd.DataFrame,
        current_price: float,
    ) -> SignalResult:
        """
        복수 기술적 지표를 OR 로직으로 분석하여 매매 신호를 생성한다.

        최소 20봉의 데이터가 필요하며, 각 조건을 독립적으로 평가한 후
        하나라도 충족되면 매매 신호를 발생시킨다.

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            df: OHLCV DataFrame
            current_price: 현재가
        Returns:
            SignalResult
        """
        # ── 최소 데이터 검증 ──
        min_required = 20
        if len(df) < min_required:
            logger.debug(
                "[%s] 스캘핑 분석 스킵 - 데이터 부족 (필요: %d, 현재: %d)",
                market,
                min_required,
                len(df),
            )
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                reason=f"데이터 부족 (필요: {min_required}, 현재: {len(df)})",
            )

        # ── 지표 계산 ──
        rsi_series = Indicators.calculate_rsi(df, period=self._rsi_period)
        ema_fast_series = Indicators.calculate_ema(df, period=self._ema_fast)
        ema_slow_series = Indicators.calculate_ema(df, period=self._ema_slow)
        bb_upper, _bb_middle, bb_lower = Indicators.calculate_bollinger_bands(
            df, period=self._bb_period, std=self._bb_std_dev
        )
        volume_sma = Indicators.calculate_volume_ma(df, period=20)

        # NaN 방어: 핵심 지표가 계산 불가하면 HOLD
        current_rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else float("nan")
        if pd.isna(current_rsi):
            logger.warning("[%s] 스캘핑 지표 계산 불가 (NaN). HOLD 반환.", market)
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                reason="지표 계산 불가 (NaN)",
            )

        # ── 매수 조건 평가 ──
        buy_reasons: List[str] = []
        buy_reasons = self._check_buy_conditions(
            df, rsi_series, ema_fast_series, ema_slow_series,
            bb_lower, volume_sma,
        )

        # ── 매도 조건 평가 ──
        sell_reasons: List[str] = []
        sell_reasons = self._check_sell_conditions(
            df, rsi_series, ema_fast_series, ema_slow_series,
            bb_upper, volume_sma,
        )

        # ── 메타데이터 구성 ──
        prev_rsi = rsi_series.iloc[-2] if len(rsi_series) >= 2 else float("nan")
        metadata = {
            "rsi": round(float(current_rsi), 2),
            "prev_rsi": round(float(prev_rsi), 2) if not pd.isna(prev_rsi) else None,
            "ema_fast": round(float(ema_fast_series.iloc[-1]), 2) if not pd.isna(ema_fast_series.iloc[-1]) else None,
            "ema_slow": round(float(ema_slow_series.iloc[-1]), 2) if not pd.isna(ema_slow_series.iloc[-1]) else None,
            "current_price": current_price,
        }

        # ── 신호 결정: 매수 우선 (OR 로직) ──
        if buy_reasons:
            count = len(buy_reasons)
            confidence = _CONFIDENCE_MAP.get(count, 0.95)
            reason = f"스캘핑 매수 ({count}개 조건): " + " | ".join(buy_reasons)
            metadata["buy_conditions"] = buy_reasons
            metadata["condition_count"] = count

            logger.info(
                "[%s] 매수 신호 - %s (confidence=%.2f)",
                market, reason, confidence,
            )
            return SignalResult(
                signal=Signal.BUY,
                market=market,
                confidence=confidence,
                reason=reason,
                metadata=metadata,
            )

        if sell_reasons:
            count = len(sell_reasons)
            confidence = _CONFIDENCE_MAP.get(count, 0.95)
            reason = f"스캘핑 매도 ({count}개 조건): " + " | ".join(sell_reasons)
            metadata["sell_conditions"] = sell_reasons
            metadata["condition_count"] = count

            logger.info(
                "[%s] 매도 신호 - %s (confidence=%.2f)",
                market, reason, confidence,
            )
            return SignalResult(
                signal=Signal.SELL,
                market=market,
                confidence=confidence,
                reason=reason,
                metadata=metadata,
            )

        # ── 관망 ──
        logger.info(
            "[%s] 스캘핑 관망: RSI=%.1f, 매수/매도 조건 미충족",
            market,
            current_rsi,
        )
        return SignalResult(
            signal=Signal.HOLD,
            market=market,
            confidence=0.0,
            reason=f"스캘핑 관망: 매수/매도 조건 미충족 (RSI={current_rsi:.1f})",
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    #  매수 조건 평가
    # ------------------------------------------------------------------ #

    def _check_buy_conditions(
        self,
        df: pd.DataFrame,
        rsi_series: pd.Series,
        ema_fast: pd.Series,
        ema_slow: pd.Series,
        bb_lower: pd.Series,
        volume_sma: pd.Series,
    ) -> List[str]:
        """매수 조건을 개별 평가하여 충족된 조건의 사유 리스트를 반환한다."""
        reasons: List[str] = []

        # 1. RSI 과매도 반등: 최근 3봉 내 RSI < oversold 이후 상승 전환
        if len(rsi_series) >= 4:
            recent_rsi = rsi_series.iloc[-3:]  # 최근 3봉
            was_oversold = any(
                not pd.isna(v) and v < self._rsi_oversold
                for v in recent_rsi.values
            )
            curr_rsi = rsi_series.iloc[-1]
            prev_rsi = rsi_series.iloc[-2]
            if was_oversold and not pd.isna(curr_rsi) and not pd.isna(prev_rsi):
                if curr_rsi > prev_rsi:
                    reasons.append(
                        f"RSI 과매도 반등 (RSI: {prev_rsi:.1f}→{curr_rsi:.1f})"
                    )

        # 2. EMA 골든크로스: 단기 EMA가 장기 EMA를 상향 돌파
        if len(ema_fast) >= 2 and len(ema_slow) >= 2:
            curr_fast = ema_fast.iloc[-1]
            prev_fast = ema_fast.iloc[-2]
            curr_slow = ema_slow.iloc[-1]
            prev_slow = ema_slow.iloc[-2]
            if (
                not pd.isna(curr_fast) and not pd.isna(prev_fast)
                and not pd.isna(curr_slow) and not pd.isna(prev_slow)
            ):
                if curr_fast > curr_slow and prev_fast <= prev_slow:
                    reasons.append(
                        f"EMA 골든크로스 ({self._ema_fast}/{self._ema_slow})"
                    )

        # 3. 볼린저 하단 반등: 최근 2봉 내 하단 터치 후 가격 상승
        if len(bb_lower) >= 2 and len(df) >= 2:
            for i in [-2, -1]:
                low_val = df["low"].iloc[i]
                lower_val = bb_lower.iloc[i]
                if not pd.isna(low_val) and not pd.isna(lower_val):
                    if low_val <= lower_val:
                        # 하단 터치 확인 → 현재 종가가 이전 종가보다 높은지
                        curr_close = df["close"].iloc[-1]
                        prev_close = df["close"].iloc[-2]
                        if (
                            not pd.isna(curr_close) and not pd.isna(prev_close)
                            and curr_close > prev_close
                        ):
                            reasons.append("볼린저 하단 반등")
                            break

        # 4. 거래량 급증 + 양봉: 거래량 > surge_ratio * SMA, 종가 > 시가
        if len(volume_sma) >= 1 and len(df) >= 1:
            curr_volume = df["volume"].iloc[-1]
            curr_vol_sma = volume_sma.iloc[-1]
            curr_close = df["close"].iloc[-1]
            curr_open = df["open"].iloc[-1]
            if (
                not pd.isna(curr_volume) and not pd.isna(curr_vol_sma)
                and not pd.isna(curr_close) and not pd.isna(curr_open)
                and curr_vol_sma > 0
            ):
                if (
                    curr_volume > self._volume_surge_ratio * curr_vol_sma
                    and curr_close > curr_open
                ):
                    ratio = curr_volume / curr_vol_sma
                    reasons.append(
                        f"거래량 급증 양봉 (vol ratio: {ratio:.1f}x)"
                    )

        return reasons

    # ------------------------------------------------------------------ #
    #  매도 조건 평가
    # ------------------------------------------------------------------ #

    def _check_sell_conditions(
        self,
        df: pd.DataFrame,
        rsi_series: pd.Series,
        ema_fast: pd.Series,
        ema_slow: pd.Series,
        bb_upper: pd.Series,
        volume_sma: pd.Series,
    ) -> List[str]:
        """매도 조건을 개별 평가하여 충족된 조건의 사유 리스트를 반환한다."""
        reasons: List[str] = []

        # 1. RSI 과매수 하락: 최근 3봉 내 RSI > overbought 이후 하락 전환
        if len(rsi_series) >= 4:
            recent_rsi = rsi_series.iloc[-3:]  # 최근 3봉
            was_overbought = any(
                not pd.isna(v) and v > self._rsi_overbought
                for v in recent_rsi.values
            )
            curr_rsi = rsi_series.iloc[-1]
            prev_rsi = rsi_series.iloc[-2]
            if was_overbought and not pd.isna(curr_rsi) and not pd.isna(prev_rsi):
                if curr_rsi < prev_rsi:
                    reasons.append(
                        f"RSI 과매수 하락 (RSI: {prev_rsi:.1f}→{curr_rsi:.1f})"
                    )

        # 2. EMA 데드크로스: 단기 EMA가 장기 EMA를 하향 돌파
        if len(ema_fast) >= 2 and len(ema_slow) >= 2:
            curr_fast = ema_fast.iloc[-1]
            prev_fast = ema_fast.iloc[-2]
            curr_slow = ema_slow.iloc[-1]
            prev_slow = ema_slow.iloc[-2]
            if (
                not pd.isna(curr_fast) and not pd.isna(prev_fast)
                and not pd.isna(curr_slow) and not pd.isna(prev_slow)
            ):
                if curr_fast < curr_slow and prev_fast >= prev_slow:
                    reasons.append(
                        f"EMA 데드크로스 ({self._ema_fast}/{self._ema_slow})"
                    )

        # 3. 볼린저 상단 이탈: 최근 2봉 내 상단 터치 후 가격 하락
        if len(bb_upper) >= 2 and len(df) >= 2:
            for i in [-2, -1]:
                high_val = df["high"].iloc[i]
                upper_val = bb_upper.iloc[i]
                if not pd.isna(high_val) and not pd.isna(upper_val):
                    if high_val >= upper_val:
                        # 상단 터치 확인 → 현재 종가가 이전 종가보다 낮은지
                        curr_close = df["close"].iloc[-1]
                        prev_close = df["close"].iloc[-2]
                        if (
                            not pd.isna(curr_close) and not pd.isna(prev_close)
                            and curr_close < prev_close
                        ):
                            reasons.append("볼린저 상단 이탈")
                            break

        # 4. 거래량 급증 + 음봉: 거래량 > surge_ratio * SMA, 종가 < 시가
        if len(volume_sma) >= 1 and len(df) >= 1:
            curr_volume = df["volume"].iloc[-1]
            curr_vol_sma = volume_sma.iloc[-1]
            curr_close = df["close"].iloc[-1]
            curr_open = df["open"].iloc[-1]
            if (
                not pd.isna(curr_volume) and not pd.isna(curr_vol_sma)
                and not pd.isna(curr_close) and not pd.isna(curr_open)
                and curr_vol_sma > 0
            ):
                if (
                    curr_volume > self._volume_surge_ratio * curr_vol_sma
                    and curr_close < curr_open
                ):
                    ratio = curr_volume / curr_vol_sma
                    reasons.append(
                        f"거래량 급증 음봉 (vol ratio: {ratio:.1f}x)"
                    )

        return reasons
