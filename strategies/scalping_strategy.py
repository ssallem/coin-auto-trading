"""
스캘핑 전략

빈번한 매매를 통해 소폭이지만 꾸준한 수익을 추구하는 단타 전략.
OR 로직 기반으로 여러 독립적인 진입 신호 중 하나라도 충족되면 매매를 실행한다.

매수 조건 (하나만 충족해도 매수):
  1. RSI 과매도: RSI < oversold 자체가 매수 신호
  2. EMA 골든크로스 또는 갭 축소: 상향 돌파 또는 하락 갭이 40% 이상 축소
  3. 볼린저 하단 터치: 현재 봉의 저가가 하단 밴드 이하
  4. 거래량 급증: 양봉 1.5x 이상 또는 음봉 2.25x 이상 (패닉셀 감지)
  5. 연속 하락 둔화: 3봉 연속 음봉 + 하락폭 30% 이상 축소

매도 조건 (하나만 충족해도 매도):
  1. RSI 과매수: RSI > overbought 자체가 매도 신호
  2. EMA 데드크로스 또는 갭 축소: 하향 돌파 또는 상승 갭이 40% 이상 축소
  3. 볼린저 상단 터치: 현재 봉의 고가가 상단 밴드 이상
  4. 거래량 급증: 음봉 1.5x 이상 또는 양봉 2.25x 이상 (차익실현 감지)
  5. 연속 상승 둔화: 3봉 연속 양봉 + 상승폭 30% 이상 축소

확신도:
  - 충족 조건 1개: 0.35
  - 충족 조건 2개: 0.55
  - 충족 조건 3개: 0.75
  - 충족 조건 4개: 0.85
  - 충족 조건 5개: 0.95
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
    1: 0.35,
    2: 0.55,
    3: 0.75,
    4: 0.85,
    5: 0.95,
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

        # ── 신호 결정: 매수/매도 동시 충족 시 reasons 수 비교 ──
        if buy_reasons and sell_reasons:
            if len(buy_reasons) > len(sell_reasons):
                # 매수 조건이 더 많으면 매수 (최소 2개 필터 적용)
                if len(buy_reasons) < 2:
                    logger.info(
                        "[%s] 매수 조건 부족 (최소 2개 필요, 현재 %d개). HOLD.",
                        market, len(buy_reasons),
                    )
                    return SignalResult(
                        signal=Signal.HOLD,
                        market=market,
                        confidence=0.0,
                        reason=f"매수/매도 동시 충족, 매수 조건 부족 ({len(buy_reasons)}개 < 2)",
                        metadata=metadata,
                    )
                count = len(buy_reasons)
                confidence = _CONFIDENCE_MAP.get(count, 0.95)
                reason = f"스캘핑 매수 ({count}개 조건): " + " | ".join(buy_reasons)
                metadata["buy_conditions"] = buy_reasons
                metadata["condition_count"] = count
                logger.info(
                    "[%s] 매수 신호 (매수/매도 동시, 매수 우세) - %s (confidence=%.2f)",
                    market, reason, confidence,
                )
                return SignalResult(
                    signal=Signal.BUY,
                    market=market,
                    confidence=confidence,
                    reason=reason,
                    metadata=metadata,
                )
            elif len(sell_reasons) > len(buy_reasons):
                # 매도 조건이 더 많으면 매도 (매도는 최소 조건 필터 없음)
                count = len(sell_reasons)
                confidence = _CONFIDENCE_MAP.get(count, 0.95)
                reason = f"스캘핑 매도 ({count}개 조건): " + " | ".join(sell_reasons)
                metadata["sell_conditions"] = sell_reasons
                metadata["condition_count"] = count
                logger.info(
                    "[%s] 매도 신호 (매수/매도 동시, 매도 우세) - %s (confidence=%.2f)",
                    market, reason, confidence,
                )
                return SignalResult(
                    signal=Signal.SELL,
                    market=market,
                    confidence=confidence,
                    reason=reason,
                    metadata=metadata,
                )
            else:
                # 동수이면 HOLD
                logger.info(
                    "[%s] 매수/매도 동시 충족 동수 (%d개). HOLD.",
                    market, len(buy_reasons),
                )
                return SignalResult(
                    signal=Signal.HOLD,
                    market=market,
                    confidence=0.0,
                    reason=f"매수/매도 동시 충족 동수 ({len(buy_reasons)}개)",
                    metadata=metadata,
                )

        if buy_reasons:
            # 매수 최소 2개 조건 필터
            if len(buy_reasons) < 2:
                logger.info(
                    "[%s] 매수 조건 부족 (최소 2개 필요, 현재 %d개). HOLD.",
                    market, len(buy_reasons),
                )
                return SignalResult(
                    signal=Signal.HOLD,
                    market=market,
                    confidence=0.0,
                    reason=f"매수 조건 부족 ({len(buy_reasons)}개 < 2)",
                    metadata=metadata,
                )
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

        # 1. RSI 과매도: RSI < oversold 자체가 매수 신호
        if len(rsi_series) >= 1:
            curr_rsi = rsi_series.iloc[-1]
            if not pd.isna(curr_rsi) and curr_rsi < self._rsi_oversold:
                reasons.append(
                    f"RSI 과매도 ({curr_rsi:.1f} < {self._rsi_oversold})"
                )

        # 2. EMA 골든크로스 또는 갭 축소
        if len(ema_fast) >= 2 and len(ema_slow) >= 2:
            curr_fast = ema_fast.iloc[-1]
            prev_fast = ema_fast.iloc[-2]
            curr_slow = ema_slow.iloc[-1]
            prev_slow = ema_slow.iloc[-2]
            if (
                not pd.isna(curr_fast) and not pd.isna(prev_fast)
                and not pd.isna(curr_slow) and not pd.isna(prev_slow)
            ):
                # (a) 기존 골든크로스
                if curr_fast > curr_slow and prev_fast <= prev_slow:
                    reasons.append(
                        f"EMA 골든크로스 ({self._ema_fast}/{self._ema_slow})"
                    )
                # (b) 갭 축소: fast < slow이지만 갭이 40% 이상 축소
                elif curr_fast < curr_slow and curr_slow > 0 and prev_slow > 0:
                    prev_gap = (prev_slow - prev_fast) / prev_slow * 100
                    curr_gap = (curr_slow - curr_fast) / curr_slow * 100
                    if prev_gap > 0.1 and curr_gap < prev_gap * 0.6:
                        reasons.append(
                            f"EMA 갭 축소 ({prev_gap:.2f}%→{curr_gap:.2f}%)"
                        )

        # 3. 볼린저 하단 터치: 현재 봉의 저가가 하단 밴드 이하
        if len(bb_lower) >= 1 and len(df) >= 1:
            curr_low = df["low"].iloc[-1]
            curr_lower = bb_lower.iloc[-1]
            if not pd.isna(curr_low) and not pd.isna(curr_lower) and curr_low <= curr_lower:
                pct_below = (curr_lower - curr_low) / curr_lower * 100 if curr_lower > 0 else 0
                reasons.append(
                    f"볼린저 하단 터치 (이탈: {pct_below:.1f}%)"
                )

        # 4. 거래량 급증: 양봉 1.5x 또는 음봉 2.25x 이상
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
                ratio = curr_volume / curr_vol_sma
                if curr_close > curr_open:
                    if ratio > self._volume_surge_ratio:
                        reasons.append(
                            f"거래량 급증 양봉 (vol: {ratio:.1f}x)"
                        )
                else:
                    if ratio > self._volume_surge_ratio * 1.5:
                        reasons.append(
                            f"거래량 급증 음봉 (vol: {ratio:.1f}x, 패닉셀 감지)"
                        )

        # 5. 연속 하락 둔화: 3봉 연속 음봉 + 하락폭 30% 이상 축소
        if len(df) >= 4:
            is_bearish = [
                df["close"].iloc[i] < df["open"].iloc[i] for i in [-3, -2, -1]
            ]
            if all(is_bearish):
                drop_prev = abs(df["close"].iloc[-2] - df["open"].iloc[-2])
                drop_curr = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
                if drop_prev > 0 and drop_curr < drop_prev * 0.7:
                    open_3 = df["open"].iloc[-3]
                    if open_3 > 0:
                        total_drop_pct = (
                            (df["close"].iloc[-1] - open_3)
                            / open_3 * 100
                        )
                        reasons.append(
                            f"연속 하락 둔화 (3봉 음봉, 총 {total_drop_pct:.1f}%)"
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

        # 1. RSI 과매수: RSI > overbought 자체가 매도 신호
        if len(rsi_series) >= 1:
            curr_rsi = rsi_series.iloc[-1]
            if not pd.isna(curr_rsi) and curr_rsi > self._rsi_overbought:
                reasons.append(
                    f"RSI 과매수 ({curr_rsi:.1f} > {self._rsi_overbought})"
                )

        # 2. EMA 데드크로스 또는 갭 축소 (상승 둔화)
        if len(ema_fast) >= 2 and len(ema_slow) >= 2:
            curr_fast = ema_fast.iloc[-1]
            prev_fast = ema_fast.iloc[-2]
            curr_slow = ema_slow.iloc[-1]
            prev_slow = ema_slow.iloc[-2]
            if (
                not pd.isna(curr_fast) and not pd.isna(prev_fast)
                and not pd.isna(curr_slow) and not pd.isna(prev_slow)
            ):
                # (a) 기존 데드크로스
                if curr_fast < curr_slow and prev_fast >= prev_slow:
                    reasons.append(
                        f"EMA 데드크로스 ({self._ema_fast}/{self._ema_slow})"
                    )
                # (b) 갭 축소: fast > slow이지만 갭이 40% 이상 축소 (상승 둔화)
                elif curr_fast > curr_slow and curr_slow > 0 and prev_slow > 0:
                    prev_gap = (prev_fast - prev_slow) / prev_slow * 100
                    curr_gap = (curr_fast - curr_slow) / curr_slow * 100
                    if prev_gap > 0.1 and curr_gap < prev_gap * 0.6:
                        reasons.append(
                            f"EMA 갭 축소 ({prev_gap:.2f}%→{curr_gap:.2f}%)"
                        )

        # 3. 볼린저 상단 터치: 현재 봉의 고가가 상단 밴드 이상
        if len(bb_upper) >= 1 and len(df) >= 1:
            curr_high = df["high"].iloc[-1]
            curr_upper = bb_upper.iloc[-1]
            if not pd.isna(curr_high) and not pd.isna(curr_upper) and curr_high >= curr_upper:
                pct_above = (curr_high - curr_upper) / curr_upper * 100 if curr_upper > 0 else 0
                reasons.append(
                    f"볼린저 상단 터치 (이탈: {pct_above:.1f}%)"
                )

        # 4. 거래량 급증: 음봉 1.5x 또는 양봉 2.25x 이상 (차익실현 감지)
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
                ratio = curr_volume / curr_vol_sma
                if curr_close < curr_open:
                    if ratio > self._volume_surge_ratio:
                        reasons.append(
                            f"거래량 급증 음봉 (vol: {ratio:.1f}x)"
                        )
                else:
                    if ratio > self._volume_surge_ratio * 1.5:
                        reasons.append(
                            f"거래량 급증 양봉 (vol: {ratio:.1f}x, 차익실현 감지)"
                        )

        # 5. 연속 상승 둔화: 3봉 연속 양봉 + 상승폭 30% 이상 축소
        if len(df) >= 4:
            is_bullish = [
                df["close"].iloc[i] > df["open"].iloc[i] for i in [-3, -2, -1]
            ]
            if all(is_bullish):
                rise_prev = abs(df["close"].iloc[-2] - df["open"].iloc[-2])
                rise_curr = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
                if rise_prev > 0 and rise_curr < rise_prev * 0.7:
                    open_3 = df["open"].iloc[-3]
                    if open_3 > 0:
                        total_rise_pct = (
                            (df["close"].iloc[-1] - open_3)
                            / open_3 * 100
                        )
                        reasons.append(
                            f"연속 상승 둔화 (3봉 양봉, 총 +{total_rise_pct:.1f}%)"
                        )

        return reasons
