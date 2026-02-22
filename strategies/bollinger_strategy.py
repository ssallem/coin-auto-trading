"""
볼린저 밴드(Bollinger Bands) 전략

볼린저 밴드의 상단/하단 이탈을 기반으로 평균 회귀(mean reversion) 매매를 수행한다.

매매 규칙:
  - 매수: 가격이 하단 밴드 아래로 이탈 (과매도 → 반등 기대)
  - 매도: 가격이 상단 밴드 위로 이탈 (과매수 → 하락 기대)
  - 관망: 밴드 내부에서 움직이는 경우

확신도(confidence):
  - 밴드 이탈 정도에 비례하여 계산한다.
  - 기본 확신도 0.5에서 시작하여, 이탈 거리가 밴드폭 대비 클수록 1.0에 접근한다.

메타데이터:
  - %B 지표를 포함한다: %B = (현재가 - 하단밴드) / (상단밴드 - 하단밴드)
    - %B < 0: 하단 밴드 이탈
    - %B > 1: 상단 밴드 이탈
    - 0 ~ 1: 밴드 내부
"""

from __future__ import annotations

import pandas as pd

from data.indicators import Indicators
from strategies.base_strategy import BaseStrategy, Signal, SignalResult
from utils.logger import get_logger

logger = get_logger(__name__)


class BollingerStrategy(BaseStrategy):
    """
    볼린저 밴드 매매 전략

    볼린저 밴드의 상하단 이탈을 감지하여 평균 회귀 매매를 수행한다.

    Args:
        period: 볼린저 밴드 이동평균 기간 (기본 20)
        std_dev: 표준편차 배수 (기본 2.0)
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        self._period = period
        self._std_dev = std_dev
        logger.info(
            "볼린저 밴드 전략 초기화: period=%d, std_dev=%.1f",
            self._period,
            self._std_dev,
        )

    # ------------------------------------------------------------------ #
    #  BaseStrategy 인터페이스
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "bollinger"

    def analyze(
        self,
        market: str,
        df: pd.DataFrame,
        current_price: float,
    ) -> SignalResult:
        """
        볼린저 밴드를 분석하여 매매 신호를 생성한다.

        1. 데이터가 부족하면 HOLD를 반환한다.
        2. 현재가 <= 하단밴드 → BUY (평균 회귀 기대)
        3. 현재가 >= 상단밴드 → SELL (과열 판단)
        4. 밴드 내부 → HOLD

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            df: OHLCV DataFrame
            current_price: 현재가
        Returns:
            SignalResult
        """
        # 최소 데이터 검증: 볼린저 밴드 계산에 period 이상 필요
        min_required = self._period + 1
        if len(df) < min_required:
            logger.debug(
                "[%s] 볼린저 분석 스킵 - 데이터 부족 (필요: %d, 현재: %d)",
                market,
                min_required,
                len(df),
            )
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                reason=f"데이터 부족 (필요: {min_required}, 현재: {len(df)})",
            )

        # 볼린저 밴드 계산
        upper, middle, lower = Indicators.bollinger_bands(
            df, period=self._period, std_dev=self._std_dev
        )

        curr_upper = upper.iloc[-1]
        curr_middle = middle.iloc[-1]
        curr_lower = lower.iloc[-1]
        band_width = curr_upper - curr_lower

        # NaN 방어
        if pd.isna(curr_upper) or pd.isna(curr_lower):
            logger.warning("[%s] 볼린저 밴드 값에 NaN 존재. HOLD 반환.", market)
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                reason="볼린저 밴드 계산 불가 (NaN)",
            )

        # %B 지표 계산: (현재가 - 하단) / (상단 - 하단)
        # %B < 0 → 하단 이탈, %B > 1 → 상단 이탈, 0~1 → 밴드 내부
        percent_b = (
            (current_price - curr_lower) / band_width
            if band_width > 0
            else 0.5
        )

        # 메타데이터 구성
        metadata = {
            "upper_band": round(float(curr_upper), 2),
            "middle_band": round(float(curr_middle), 2),
            "lower_band": round(float(curr_lower), 2),
            "band_width": round(float(band_width), 2),
            "percent_b": round(float(percent_b), 4),
            "current_price": current_price,
            "period": self._period,
            "std_dev": self._std_dev,
        }

        # ── 매수: 현재가가 하단 밴드 이하 ──
        if current_price <= curr_lower:
            # 하단밴드와의 이탈 거리 대비 밴드폭 비율로 확신도 계산
            distance_pct = (
                (curr_lower - current_price) / band_width if band_width > 0 else 0.0
            )
            confidence = min(0.5 + distance_pct, 1.0)
            reason = (
                f"하단밴드 이탈: 현재가 {current_price:,.0f}"
                f" <= 하단 {curr_lower:,.0f}"
                f" (%B={percent_b:.2f})"
            )
            logger.info("[%s] 매수 신호 - %s (confidence=%.3f)", market, reason, confidence)
            return SignalResult(
                signal=Signal.BUY,
                market=market,
                confidence=round(confidence, 3),
                reason=reason,
                metadata=metadata,
            )

        # ── 매도: 현재가가 상단 밴드 이상 ──
        if current_price >= curr_upper:
            distance_pct = (
                (current_price - curr_upper) / band_width if band_width > 0 else 0.0
            )
            confidence = min(0.5 + distance_pct, 1.0)
            reason = (
                f"상단밴드 이탈: 현재가 {current_price:,.0f}"
                f" >= 상단 {curr_upper:,.0f}"
                f" (%B={percent_b:.2f})"
            )
            logger.info("[%s] 매도 신호 - %s (confidence=%.3f)", market, reason, confidence)
            return SignalResult(
                signal=Signal.SELL,
                market=market,
                confidence=round(confidence, 3),
                reason=reason,
                metadata=metadata,
            )

        # ── 관망: 밴드 내부 ──
        logger.debug(
            "[%s] 볼린저 관망: 밴드 내 위치 %.1f%% (%%B=%.2f)",
            market,
            percent_b * 100,
            percent_b,
        )
        return SignalResult(
            signal=Signal.HOLD,
            market=market,
            confidence=0.0,
            reason=f"밴드 내부: 위치 {percent_b:.1%} (%B={percent_b:.2f})",
            metadata=metadata,
        )
