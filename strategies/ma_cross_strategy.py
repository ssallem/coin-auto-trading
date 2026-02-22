"""
이동평균 교차(MA Cross) 전략

단기 이동평균과 장기 이동평균의 교차를 기반으로 매매한다.

매매 규칙:
  - 매수: 단기 MA가 장기 MA를 상향 돌파 (골든 크로스)
  - 매도: 단기 MA가 장기 MA를 하향 돌파 (데드 크로스)
  - 관망: 교차가 발생하지 않은 경우

크로스 감지 로직:
  - 직전 봉에서는 반대 관계였는데 현재 봉에서 교차가 발생한 경우에만 신호를 생성한다.
  - 골든 크로스: prev(단기 <= 장기) AND curr(단기 > 장기)
  - 데드 크로스: prev(단기 >= 장기) AND curr(단기 < 장기)

확신도(confidence):
  - 이격도(두 MA의 차이 비율)에 비례하여 계산한다.
  - 이격도가 1% 이상이면 확신도 1.0 (상한 클램프)
"""

from __future__ import annotations

import pandas as pd

from data.indicators import Indicators
from strategies.base_strategy import BaseStrategy, Signal, SignalResult
from utils.logger import get_logger

logger = get_logger(__name__)


class MACrossStrategy(BaseStrategy):
    """
    이동평균 교차 매매 전략

    단기/장기 이동평균의 교차 시점을 포착하여 매매 신호를 생성한다.
    SMA 또는 EMA를 선택적으로 사용할 수 있다.

    Args:
        short_period: 단기 이동평균 기간 (기본 5)
        long_period: 장기 이동평균 기간 (기본 20)
        use_ema: True면 EMA, False면 SMA 사용 (기본 False)
    """

    def __init__(
        self,
        short_period: int = 5,
        long_period: int = 20,
        use_ema: bool = False,
    ) -> None:
        self._short_period = short_period
        self._long_period = long_period
        self._use_ema = use_ema
        # 내부적으로 Indicators.moving_average()에 전달할 ma_type 문자열
        self._ma_type = "EMA" if use_ema else "SMA"
        logger.info(
            "MA Cross 전략 초기화: short=%d, long=%d, type=%s",
            self._short_period,
            self._long_period,
            self._ma_type,
        )

    # ------------------------------------------------------------------ #
    #  BaseStrategy 인터페이스
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "ma_cross"

    def analyze(
        self,
        market: str,
        df: pd.DataFrame,
        current_price: float,
    ) -> SignalResult:
        """
        이동평균 교차를 분석하여 매매 신호를 생성한다.

        1. 데이터가 부족하면 HOLD를 반환한다.
        2. 직전 봉 대비 현재 봉에서 교차가 발생했는지 확인한다.
        3. 골든 크로스 → BUY, 데드 크로스 → SELL, 그 외 → HOLD

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            df: OHLCV DataFrame
            current_price: 현재가
        Returns:
            SignalResult
        """
        # 최소 데이터 검증: 장기 MA 계산 + 교차 감지를 위해 최소 2봉 추가
        min_required = self._long_period + 2
        if len(df) < min_required:
            logger.debug(
                "[%s] MA Cross 분석 스킵 - 데이터 부족 (필요: %d, 현재: %d)",
                market,
                min_required,
                len(df),
            )
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                reason=f"데이터 부족 (필요: {min_required}, 현재: {len(df)})",
            )

        # 이동평균 계산 (SMA 또는 EMA)
        short_ma = Indicators.moving_average(df, self._short_period, self._ma_type)
        long_ma = Indicators.moving_average(df, self._long_period, self._ma_type)

        # NaN 방어
        if pd.isna(short_ma.iloc[-1]) or pd.isna(long_ma.iloc[-1]):
            logger.warning("[%s] MA 값에 NaN 존재. HOLD 반환.", market)
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                reason="이동평균 계산 불가 (NaN)",
            )

        # 현재 봉과 직전 봉의 단기-장기 MA 차이
        curr_diff = short_ma.iloc[-1] - long_ma.iloc[-1]
        prev_diff = short_ma.iloc[-2] - long_ma.iloc[-2]

        # 이격도 (%) = |단기MA - 장기MA| / 장기MA * 100
        long_ma_val = long_ma.iloc[-1]
        spread_pct = (
            abs(curr_diff) / long_ma_val * 100 if long_ma_val != 0 else 0.0
        )

        # 메타데이터 구성
        metadata = {
            "short_ma": round(float(short_ma.iloc[-1]), 2),
            "long_ma": round(float(long_ma.iloc[-1]), 2),
            "short_period": self._short_period,
            "long_period": self._long_period,
            "ma_type": self._ma_type,
            "diff": round(float(curr_diff), 2),
            "spread_pct": round(spread_pct, 4),
            "current_price": current_price,
        }

        # ── 골든 크로스: 직전에 단기 <= 장기였고, 현재 단기 > 장기 ──
        if prev_diff <= 0 and curr_diff > 0:
            # 이격도가 1% 이상이면 확신도 1.0으로 클램프
            confidence = min(spread_pct / 1.0, 1.0)
            reason = (
                f"골든 크로스: {self._ma_type}{self._short_period}"
                f"({short_ma.iloc[-1]:,.0f}) > "
                f"{self._ma_type}{self._long_period}"
                f"({long_ma.iloc[-1]:,.0f}), "
                f"이격도 {spread_pct:.2f}%"
            )
            logger.info("[%s] 매수 신호 - %s (confidence=%.3f)", market, reason, confidence)
            return SignalResult(
                signal=Signal.BUY,
                market=market,
                confidence=round(confidence, 3),
                reason=reason,
                metadata=metadata,
            )

        # ── 데드 크로스: 직전에 단기 >= 장기였고, 현재 단기 < 장기 ──
        if prev_diff >= 0 and curr_diff < 0:
            confidence = min(spread_pct / 1.0, 1.0)
            reason = (
                f"데드 크로스: {self._ma_type}{self._short_period}"
                f"({short_ma.iloc[-1]:,.0f}) < "
                f"{self._ma_type}{self._long_period}"
                f"({long_ma.iloc[-1]:,.0f}), "
                f"이격도 {spread_pct:.2f}%"
            )
            logger.info("[%s] 매도 신호 - %s (confidence=%.3f)", market, reason, confidence)
            return SignalResult(
                signal=Signal.SELL,
                market=market,
                confidence=round(confidence, 3),
                reason=reason,
                metadata=metadata,
            )

        # ── 관망: 교차 발생하지 않음 ──
        trend = "상승세" if curr_diff > 0 else "하락세"
        logger.debug(
            "[%s] MA Cross 관망: %s (단기-장기 차이=%.2f)",
            market,
            trend,
            curr_diff,
        )
        return SignalResult(
            signal=Signal.HOLD,
            market=market,
            confidence=0.0,
            reason=f"교차 없음 ({trend}, 이격도 {spread_pct:.2f}%)",
            metadata=metadata,
        )
