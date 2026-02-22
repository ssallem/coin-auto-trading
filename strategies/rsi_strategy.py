"""
RSI 전략

RSI(Relative Strength Index) 기반 매매 전략.
과매도 구간에서 매수, 과매수 구간에서 매도한다.

매매 규칙:
  - 매수: RSI < oversold (기본 30)
  - 매도: RSI > overbought (기본 70)
  - 관망: oversold <= RSI <= overbought

확신도(confidence):
  - RSI가 극단값(0 또는 100)에 가까울수록 높은 확신도를 반환한다.
  - 매수 시: (oversold - RSI) / oversold  → RSI가 0에 가까울수록 1.0
  - 매도 시: (RSI - overbought) / (100 - overbought) → RSI가 100에 가까울수록 1.0
"""

from __future__ import annotations

import pandas as pd

from data.indicators import Indicators
from strategies.base_strategy import BaseStrategy, Signal, SignalResult
from utils.logger import get_logger

logger = get_logger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI 기반 매매 전략

    RSI가 과매도 임계값 아래로 내려가면 매수 신호를,
    과매수 임계값 위로 올라가면 매도 신호를 생성한다.

    Args:
        period: RSI 계산 기간 (기본 14)
        oversold: 과매도 기준값 (기본 30)
        overbought: 과매수 기준값 (기본 70)
    """

    def __init__(
        self,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
    ) -> None:
        self._period = period
        self._oversold = oversold
        self._overbought = overbought
        logger.info(
            "RSI 전략 초기화: period=%d, oversold=%d, overbought=%d",
            self._period,
            self._oversold,
            self._overbought,
        )

    # ------------------------------------------------------------------ #
    #  BaseStrategy 인터페이스
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "rsi"

    def analyze(
        self,
        market: str,
        df: pd.DataFrame,
        current_price: float,
    ) -> SignalResult:
        """
        RSI를 계산하여 매매 신호를 생성한다.

        1. 데이터가 부족하면 HOLD를 반환한다.
        2. RSI < oversold → BUY (과매도 반등 기대)
        3. RSI > overbought → SELL (과매수 하락 기대)
        4. 그 외 → HOLD (관망)

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            df: OHLCV DataFrame
            current_price: 현재가
        Returns:
            SignalResult
        """
        # 최소 데이터 검증: RSI 계산에 period + 1개 이상의 봉이 필요
        min_required = self._period + 1
        if len(df) < min_required:
            logger.debug(
                "[%s] RSI 분석 스킵 - 데이터 부족 (필요: %d, 현재: %d)",
                market,
                min_required,
                len(df),
            )
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                reason=f"데이터 부족 (필요: {min_required}, 현재: {len(df)})",
            )

        # RSI 계산
        rsi_series = Indicators.rsi(df, period=self._period)
        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]

        # NaN 방어 (초기 구간에서 발생 가능)
        if pd.isna(current_rsi):
            logger.warning("[%s] RSI 값이 NaN입니다. HOLD 반환.", market)
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                reason="RSI 계산 불가 (NaN)",
            )

        # 메타데이터 구성
        metadata = {
            "rsi": round(current_rsi, 2),
            "prev_rsi": round(float(prev_rsi), 2) if not pd.isna(prev_rsi) else None,
            "period": self._period,
            "oversold": self._oversold,
            "overbought": self._overbought,
            "current_price": current_price,
        }

        # ── 매수 신호: RSI가 과매도 영역에 진입 ──
        if current_rsi < self._oversold:
            # RSI가 0에 가까울수록 확신도가 높다
            confidence = min((self._oversold - current_rsi) / self._oversold, 1.0)
            reason = (
                f"RSI 과매도: {current_rsi:.1f} < {self._oversold} "
                f"(이전 RSI: {prev_rsi:.1f})"
            )
            logger.info("[%s] 매수 신호 - %s (confidence=%.3f)", market, reason, confidence)
            return SignalResult(
                signal=Signal.BUY,
                market=market,
                confidence=round(confidence, 3),
                reason=reason,
                metadata=metadata,
            )

        # ── 매도 신호: RSI가 과매수 영역에 진입 ──
        if current_rsi > self._overbought:
            # RSI가 100에 가까울수록 확신도가 높다
            confidence = min(
                (current_rsi - self._overbought) / (100 - self._overbought), 1.0
            )
            reason = (
                f"RSI 과매수: {current_rsi:.1f} > {self._overbought} "
                f"(이전 RSI: {prev_rsi:.1f})"
            )
            logger.info("[%s] 매도 신호 - %s (confidence=%.3f)", market, reason, confidence)
            return SignalResult(
                signal=Signal.SELL,
                market=market,
                confidence=round(confidence, 3),
                reason=reason,
                metadata=metadata,
            )

        # ── 관망: 중립 구간 ──
        logger.debug(
            "[%s] RSI 관망: %.1f (%d~%d 사이)",
            market,
            current_rsi,
            self._oversold,
            self._overbought,
        )
        return SignalResult(
            signal=Signal.HOLD,
            market=market,
            confidence=0.0,
            reason=f"RSI 중립: {current_rsi:.1f} ({self._oversold}~{self._overbought} 범위 내)",
            metadata=metadata,
        )
