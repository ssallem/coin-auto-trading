"""
Ross Cameron 스타일 매매 전략

Ross Cameron의 모멘텀 데이 트레이딩 전략을 기반으로 한 자동매매 전략.
두 가지 독립적인 전략을 조합하여 강력한 매수/매도 신호를 생성한다.

전략 A (패턴 + 볼린저 밴드):
  - RSI 과매도/과매수 구간
  - 연속 음봉/양봉 + 볼린저 밴드 터치
  - 장악형 패턴 (Engulfing)
  - 쌍바닥/쌍봉 패턴 (밴드 내부 조건 포함)

전략 B (다이버전스 + MACD):
  - 상승/하락 다이버전스 (RSI vs 가격)
  - MACD 골든/데드 크로스
  - 장악형 패턴 확인

리스크 관리:
  - 손절가: 볼린저 하단 또는 최근 저점
  - 목표가: 리스크 대비 2배 이상 리워드 (매수) / 1배 이상 (매도)
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, SignalResult
from data.indicators import Indicators
from utils.candle_patterns import CandlePatterns
from utils.divergence_detector import DivergenceDetector
from utils.logger import get_logger

logger = get_logger(__name__)


class RossCameronStrategy(BaseStrategy):
    """
    Ross Cameron 스타일 매매 전략

    패턴 인식, 기술적 지표, 다이버전스 분석을 결합하여
    강력한 매수/매도 신호를 생성하는 복합 전략.
    """

    def __init__(
        self,
        # 공통 파라미터
        rsi_period: int = 14,
        rsi_neutral_min: int = 40,
        rsi_neutral_max: int = 60,
        # 전략 A 파라미터
        enable_strategy_a: bool = True,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        double_pattern_window: int = 30,
        # 전략 B 파라미터
        enable_strategy_b: bool = True,
        divergence_window: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_period: int = 9,
        # 리스크 관리 파라미터
        buy_risk_reward_ratio: float = 2.0,
        sell_risk_reward_ratio: float = 1.0,
    ):
        """
        Args:
            rsi_period: RSI 계산 기간
            rsi_neutral_min: RSI 중립 구간 하한 (이 이상이면 변동성 부족)
            rsi_neutral_max: RSI 중립 구간 상한 (이 이하이면 변동성 부족)
            enable_strategy_a: 전략 A 활성화 여부
            bb_period: 볼린저 밴드 기간
            bb_std_dev: 볼린저 밴드 표준편차 배수
            double_pattern_window: 쌍바닥/쌍봉 탐색 윈도우
            enable_strategy_b: 전략 B 활성화 여부
            divergence_window: 다이버전스 탐색 윈도우
            macd_fast: MACD 단기 EMA 기간
            macd_slow: MACD 장기 EMA 기간
            macd_signal_period: MACD 시그널 기간
            buy_risk_reward_ratio: 매수 시 리스크 대비 리워드 비율
            sell_risk_reward_ratio: 매도 시 리스크 대비 리워드 비율
        """
        # 공통
        self._rsi_period = rsi_period
        self._rsi_neutral_min = rsi_neutral_min
        self._rsi_neutral_max = rsi_neutral_max

        # 전략 A
        self._enable_strategy_a = enable_strategy_a
        self._bb_period = bb_period
        self._bb_std_dev = bb_std_dev
        self._double_pattern_window = double_pattern_window

        # 전략 B
        self._enable_strategy_b = enable_strategy_b
        self._divergence_window = divergence_window
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal_period = macd_signal_period

        # 리스크 관리
        self._buy_risk_reward_ratio = buy_risk_reward_ratio
        self._sell_risk_reward_ratio = sell_risk_reward_ratio

        logger.info(
            f"[{self.name}] 전략 초기화 완료: "
            f"Strategy A={enable_strategy_a}, Strategy B={enable_strategy_b}"
        )

    @property
    def name(self) -> str:
        """전략 이름"""
        return "ross_cameron"

    def analyze(
        self,
        market: str,
        df: pd.DataFrame,
        current_price: float,
    ) -> SignalResult:
        """
        시세 데이터를 분석하여 매매 신호를 생성한다.

        분석 흐름:
        1. 데이터 검증 (최소 60개 봉 필요)
        2. 기술적 지표 확인 및 계산
        3. RSI 중립 구간 필터링
        4. 전략 A 매수/매도 체크
        5. 전략 B 매수/매도 체크
        6. 신호 통합 및 확신도 계산
        7. 손절/목표가 계산

        Args:
            market: 마켓 코드
            df: OHLCV DataFrame
            current_price: 현재가

        Returns:
            SignalResult
        """
        # 1. 데이터 검증
        if len(df) < 60:
            logger.debug(
                f"[{self.name}] {market} 데이터 부족: {len(df)}개 < 60개"
            )
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                confidence=0.0,
                reason="데이터 부족 (60개 미만)",
                metadata={"data_length": len(df)}
            )

        # 2. 지표 확인 및 계산
        df = self._ensure_indicators(df)

        # 필수 지표 존재 확인
        required_indicators = ['rsi', 'bb_upper', 'bb_lower', 'macd', 'macd_signal']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]

        if missing_indicators:
            logger.warning(
                f"[{self.name}] {market} 필수 지표 누락: {missing_indicators}"
            )
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                confidence=0.0,
                reason=f"필수 지표 누락: {missing_indicators}",
                metadata={"missing_indicators": missing_indicators}
            )

        # 3. 현재 RSI 값 확인
        current_rsi = df['rsi'].iloc[-1]

        if pd.isna(current_rsi):
            logger.debug(f"[{self.name}] {market} RSI 값이 NaN")
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                confidence=0.0,
                reason="RSI 값이 유효하지 않음",
                metadata={"rsi": None}
            )

        logger.debug(f"[{self.name}] {market} 현재 RSI: {current_rsi:.2f}")

        # 4. RSI 중립 구간 필터링 (변동성 부족)
        if self._rsi_neutral_min <= current_rsi <= self._rsi_neutral_max:
            logger.info(
                f"[{self.name}] {market} RSI 중립 구간 ({current_rsi:.2f}): "
                f"변동성 부족으로 관망"
            )
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                confidence=0.0,
                reason=f"RSI 중립 구간 ({current_rsi:.2f}): 변동성 부족",
                metadata={"rsi": current_rsi, "neutral_range": [self._rsi_neutral_min, self._rsi_neutral_max]}
            )

        # 5. 전략 A 체크
        strategy_a_buy = False
        strategy_a_sell = False
        strategy_a_metadata = {}

        if self._enable_strategy_a:
            strategy_a_buy, a_buy_meta = self._check_strategy_a_buy(
                df, current_rsi
            )
            strategy_a_sell, a_sell_meta = self._check_strategy_a_sell(
                df, current_rsi
            )

            strategy_a_metadata = {
                "buy": a_buy_meta if strategy_a_buy else None,
                "sell": a_sell_meta if strategy_a_sell else None,
            }

            logger.debug(
                f"[{self.name}] {market} 전략 A - "
                f"매수={strategy_a_buy}, 매도={strategy_a_sell}"
            )

        # 6. 전략 B 체크
        strategy_b_buy = False
        strategy_b_sell = False
        strategy_b_metadata = {}

        if self._enable_strategy_b:
            strategy_b_buy, b_buy_meta = self._check_strategy_b_buy(df)
            strategy_b_sell, b_sell_meta = self._check_strategy_b_sell(df)

            strategy_b_metadata = {
                "buy": b_buy_meta if strategy_b_buy else None,
                "sell": b_sell_meta if strategy_b_sell else None,
            }

            logger.debug(
                f"[{self.name}] {market} 전략 B - "
                f"매수={strategy_b_buy}, 매도={strategy_b_sell}"
            )

        # 7. 신호 결정 및 확신도 계산
        buy_signals = sum([strategy_a_buy, strategy_b_buy])
        sell_signals = sum([strategy_a_sell, strategy_b_sell])

        # 매수와 매도 신호가 동시에 나오면 HOLD (상충)
        if buy_signals > 0 and sell_signals > 0:
            logger.info(
                f"[{self.name}] {market} 신호 충돌: "
                f"매수={buy_signals}, 매도={sell_signals} - 관망"
            )
            return SignalResult(
                signal=Signal.HOLD,
                market=market,
                confidence=0.0,
                reason="매수/매도 신호 충돌",
                metadata={
                    "rsi": current_rsi,
                    "strategy_a": strategy_a_metadata,
                    "strategy_b": strategy_b_metadata,
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                }
            )

        # 매수 신호
        if buy_signals > 0:
            # 확신도: 1개 전략 = 0.6, 2개 전략 = 0.9
            confidence = 0.6 if buy_signals == 1 else 0.9

            # 손절/목표가 계산
            bb_lower_val = df['bb_lower'].iloc[-1]
            bb_upper_val = df['bb_upper'].iloc[-1]
            risk_reward = self._calculate_stop_take_profit(
                Signal.BUY, current_price, df, bb_lower_val, bb_upper_val
            )

            reason = f"매수 신호 ({buy_signals}개 전략 일치): RSI={current_rsi:.2f}"
            if strategy_a_buy:
                reason += " [전략 A: 패턴+볼린저]"
            if strategy_b_buy:
                reason += " [전략 B: 다이버전스+MACD]"

            logger.info(
                f"[{self.name}] {market} 매수 신호 발생: "
                f"확신도={confidence:.2f}, {reason}"
            )

            return SignalResult(
                signal=Signal.BUY,
                market=market,
                confidence=confidence,
                reason=reason,
                metadata={
                    "rsi": current_rsi,
                    "current_price": current_price,
                    "strategy_a": strategy_a_metadata,
                    "strategy_b": strategy_b_metadata,
                    "active_strategies": buy_signals,
                    "risk_reward": risk_reward,
                }
            )

        # 매도 신호
        if sell_signals > 0:
            # 확신도: 1개 전략 = 0.6, 2개 전략 = 0.9
            confidence = 0.6 if sell_signals == 1 else 0.9

            # 손절/목표가 계산
            bb_lower_val = df['bb_lower'].iloc[-1]
            bb_upper_val = df['bb_upper'].iloc[-1]
            risk_reward = self._calculate_stop_take_profit(
                Signal.SELL, current_price, df, bb_lower_val, bb_upper_val
            )

            reason = f"매도 신호 ({sell_signals}개 전략 일치): RSI={current_rsi:.2f}"
            if strategy_a_sell:
                reason += " [전략 A: 패턴+볼린저]"
            if strategy_b_sell:
                reason += " [전략 B: 다이버전스+MACD]"

            logger.info(
                f"[{self.name}] {market} 매도 신호 발생: "
                f"확신도={confidence:.2f}, {reason}"
            )

            return SignalResult(
                signal=Signal.SELL,
                market=market,
                confidence=confidence,
                reason=reason,
                metadata={
                    "rsi": current_rsi,
                    "current_price": current_price,
                    "strategy_a": strategy_a_metadata,
                    "strategy_b": strategy_b_metadata,
                    "active_strategies": sell_signals,
                    "risk_reward": risk_reward,
                }
            )

        # 신호 없음
        logger.debug(f"[{self.name}] {market} 신호 없음 - 관망")
        return SignalResult(
            signal=Signal.HOLD,
            market=market,
            confidence=0.0,
            reason="매매 조건 미충족",
            metadata={
                "rsi": current_rsi,
                "strategy_a": strategy_a_metadata,
                "strategy_b": strategy_b_metadata,
            }
        )

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        필수 지표가 DataFrame에 있는지 확인하고, 없으면 계산하여 추가한다.

        Args:
            df: OHLCV DataFrame

        Returns:
            지표가 추가된 DataFrame
        """
        result = df.copy()

        # RSI
        if 'rsi' not in result.columns:
            logger.debug("RSI 지표 계산 중...")
            result['rsi'] = Indicators.calculate_rsi(df, period=self._rsi_period)

        # 볼린저 밴드
        if 'bb_upper' not in result.columns or 'bb_lower' not in result.columns:
            logger.debug("볼린저 밴드 계산 중...")
            upper, middle, lower = Indicators.calculate_bollinger_bands(
                df, period=self._bb_period, std=self._bb_std_dev
            )
            result['bb_upper'] = upper
            result['bb_middle'] = middle
            result['bb_lower'] = lower

        # MACD
        if 'macd' not in result.columns or 'macd_signal' not in result.columns:
            logger.debug("MACD 계산 중...")
            macd_line, signal_line, histogram = Indicators.calculate_macd(
                df,
                fast=self._macd_fast,
                slow=self._macd_slow,
                signal=self._macd_signal_period
            )
            result['macd'] = macd_line
            result['macd_signal'] = signal_line
            result['macd_hist'] = histogram

        return result

    def _check_strategy_a_buy(
        self,
        df: pd.DataFrame,
        rsi: float,
    ) -> Tuple[bool, Dict]:
        """
        전략 A 매수 조건 체크

        조건 (4가지 모두 충족):
        1. RSI ≤ 30 (과매도)
        2. 연속 음봉 2개 이상 AND 최근 볼린저 하단 터치
        3. 장악형 양봉 패턴
        4. 쌍바닥 패턴 AND 밴드 내부 조건

        Args:
            df: OHLCV DataFrame
            rsi: 현재 RSI 값
            bb_lower_series: 볼린저 하단 Series

        Returns:
            (조건 충족 여부, 메타데이터)
        """
        metadata = {
            "rsi_oversold": False,
            "consecutive_bearish": 0,
            "band_touch": False,
            "bullish_engulfing": False,
            "double_bottom": None,
        }

        # 조건 1: RSI ≤ 30
        if rsi > 30:
            logger.debug(f"전략 A 매수: RSI 조건 불만족 ({rsi:.2f} > 30)")
            return False, metadata

        metadata["rsi_oversold"] = True
        logger.debug(f"전략 A 매수: RSI 과매도 확인 ({rsi:.2f})")

        # 조건 2: 연속 음봉 2개 이상 AND 볼린저 하단 터치
        consecutive_bearish = CandlePatterns.count_consecutive_bearish(df)
        band_touch = CandlePatterns.recent_touches_lower_band(df)

        metadata["consecutive_bearish"] = consecutive_bearish
        metadata["band_touch"] = band_touch

        if consecutive_bearish < 2 or not band_touch:
            logger.debug(
                f"전략 A 매수: 연속 음봉 또는 밴드 터치 조건 불만족 "
                f"(음봉={consecutive_bearish}, 터치={band_touch})"
            )
            return False, metadata

        logger.debug(
            f"전략 A 매수: 연속 음봉 {consecutive_bearish}개 + 하단밴드 터치 확인"
        )

        # 조건 3: 장악형 양봉
        bullish_engulfing = CandlePatterns.is_bullish_engulfing(df)
        metadata["bullish_engulfing"] = bullish_engulfing

        if not bullish_engulfing:
            logger.debug("전략 A 매수: 장악형 양봉 조건 불만족")
            return False, metadata

        logger.debug("전략 A 매수: 장악형 양봉 확인")

        # 조건 4: 쌍바닥 패턴 AND 밴드 내부
        double_bottom = CandlePatterns.find_double_bottom(
            df, window=self._double_pattern_window
        )
        metadata["double_bottom"] = double_bottom

        if double_bottom is None:
            logger.debug("전략 A 매수: 쌍바닥 패턴 불발견")
            return False, metadata

        if not double_bottom.get("inside_band", False):
            logger.debug("전략 A 매수: 쌍바닥이 밴드 외부")
            return False, metadata

        logger.debug(
            f"전략 A 매수: 쌍바닥 확인 (밴드 내부) - "
            f"저점1={double_bottom['first_low']:.2f}, "
            f"저점2={double_bottom['second_low']:.2f}"
        )

        # 모든 조건 충족
        logger.info("전략 A 매수: 모든 조건 충족!")
        return True, metadata

    def _check_strategy_a_sell(
        self,
        df: pd.DataFrame,
        rsi: float,
    ) -> Tuple[bool, Dict]:
        """
        전략 A 매도 조건 체크

        조건 (4가지 모두 충족):
        1. RSI ≥ 70 (과매수)
        2. 연속 양봉 2개 이상 AND 최근 볼린저 상단 터치
        3. 하락 장악형 패턴
        4. 쌍봉 패턴 AND 밴드 내부 조건

        Args:
            df: OHLCV DataFrame
            rsi: 현재 RSI 값
            bb_upper_series: 볼린저 상단 Series

        Returns:
            (조건 충족 여부, 메타데이터)
        """
        metadata = {
            "rsi_overbought": False,
            "consecutive_bullish": 0,
            "band_touch": False,
            "bearish_engulfing": False,
            "double_top": None,
        }

        # 조건 1: RSI ≥ 70
        if rsi < 70:
            logger.debug(f"전략 A 매도: RSI 조건 불만족 ({rsi:.2f} < 70)")
            return False, metadata

        metadata["rsi_overbought"] = True
        logger.debug(f"전략 A 매도: RSI 과매수 확인 ({rsi:.2f})")

        # 조건 2: 연속 양봉 2개 이상 AND 볼린저 상단 터치
        consecutive_bullish = CandlePatterns.count_consecutive_bullish(df)
        band_touch = CandlePatterns.recent_touches_upper_band(df)

        metadata["consecutive_bullish"] = consecutive_bullish
        metadata["band_touch"] = band_touch

        if consecutive_bullish < 2 or not band_touch:
            logger.debug(
                f"전략 A 매도: 연속 양봉 또는 밴드 터치 조건 불만족 "
                f"(양봉={consecutive_bullish}, 터치={band_touch})"
            )
            return False, metadata

        logger.debug(
            f"전략 A 매도: 연속 양봉 {consecutive_bullish}개 + 상단밴드 터치 확인"
        )

        # 조건 3: 하락 장악형
        bearish_engulfing = CandlePatterns.is_bearish_engulfing(df)
        metadata["bearish_engulfing"] = bearish_engulfing

        if not bearish_engulfing:
            logger.debug("전략 A 매도: 하락 장악형 조건 불만족")
            return False, metadata

        logger.debug("전략 A 매도: 하락 장악형 확인")

        # 조건 4: 쌍봉 패턴 AND 밴드 내부
        double_top = CandlePatterns.find_double_top(
            df, window=self._double_pattern_window
        )
        metadata["double_top"] = double_top

        if double_top is None:
            logger.debug("전략 A 매도: 쌍봉 패턴 불발견")
            return False, metadata

        if not double_top.get("inside_band", False):
            logger.debug("전략 A 매도: 쌍봉이 밴드 외부")
            return False, metadata

        logger.debug(
            f"전략 A 매도: 쌍봉 확인 (밴드 내부) - "
            f"고점1={double_top['first_high']:.2f}, "
            f"고점2={double_top['second_high']:.2f}"
        )

        # 모든 조건 충족
        logger.info("전략 A 매도: 모든 조건 충족!")
        return True, metadata

    def _check_strategy_b_buy(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, Dict]:
        """
        전략 B 매수 조건 체크

        조건 (3가지 모두 충족):
        1. 상승 다이버전스 (가격↓ RSI↑)
        2. MACD 골든 크로스
        3. 장악형 양봉 패턴

        Args:
            df: OHLCV DataFrame
            rsi_series: RSI Series
            macd_line: MACD Line Series
            signal_line: MACD Signal Series

        Returns:
            (조건 충족 여부, 메타데이터)
        """
        metadata = {
            "bullish_divergence": None,
            "macd_golden_cross": False,
            "bullish_engulfing": False,
        }

        # 조건 1: 상승 다이버전스
        divergence = DivergenceDetector.detect_bullish_divergence(
            df, window=self._divergence_window
        )
        metadata["bullish_divergence"] = divergence

        if divergence is None:
            logger.debug("전략 B 매수: 상승 다이버전스 불발견")
            return False, metadata

        logger.debug(
            f"전략 B 매수: 상승 다이버전스 확인 - "
            f"강도={divergence['strength']:.3f}"
        )

        # 조건 2: MACD 골든 크로스
        golden_cross = DivergenceDetector.is_macd_golden_cross(df, lookback=3)
        metadata["macd_golden_cross"] = golden_cross

        if not golden_cross:
            logger.debug("전략 B 매수: MACD 골든 크로스 불발견")
            return False, metadata

        logger.debug("전략 B 매수: MACD 골든 크로스 확인")

        # 조건 3: 장악형 양봉
        bullish_engulfing = CandlePatterns.is_bullish_engulfing(df)
        metadata["bullish_engulfing"] = bullish_engulfing

        if not bullish_engulfing:
            logger.debug("전략 B 매수: 장악형 양봉 조건 불만족")
            return False, metadata

        logger.debug("전략 B 매수: 장악형 양봉 확인")

        # 모든 조건 충족
        logger.info("전략 B 매수: 모든 조건 충족!")
        return True, metadata

    def _check_strategy_b_sell(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, Dict]:
        """
        전략 B 매도 조건 체크

        조건 (3가지 모두 충족):
        1. 하락 다이버전스 (가격↑ RSI↓)
        2. MACD 데드 크로스
        3. 하락 장악형 패턴

        Args:
            df: OHLCV DataFrame
            rsi_series: RSI Series
            macd_line: MACD Line Series
            signal_line: MACD Signal Series

        Returns:
            (조건 충족 여부, 메타데이터)
        """
        metadata = {
            "bearish_divergence": None,
            "macd_death_cross": False,
            "bearish_engulfing": False,
        }

        # 조건 1: 하락 다이버전스
        divergence = DivergenceDetector.detect_bearish_divergence(
            df, window=self._divergence_window
        )
        metadata["bearish_divergence"] = divergence

        if divergence is None:
            logger.debug("전략 B 매도: 하락 다이버전스 불발견")
            return False, metadata

        logger.debug(
            f"전략 B 매도: 하락 다이버전스 확인 - "
            f"강도={divergence['strength']:.3f}"
        )

        # 조건 2: MACD 데드 크로스
        death_cross = DivergenceDetector.is_macd_death_cross(df, lookback=3)
        metadata["macd_death_cross"] = death_cross

        if not death_cross:
            logger.debug("전략 B 매도: MACD 데드 크로스 불발견")
            return False, metadata

        logger.debug("전략 B 매도: MACD 데드 크로스 확인")

        # 조건 3: 하락 장악형
        bearish_engulfing = CandlePatterns.is_bearish_engulfing(df)
        metadata["bearish_engulfing"] = bearish_engulfing

        if not bearish_engulfing:
            logger.debug("전략 B 매도: 하락 장악형 조건 불만족")
            return False, metadata

        logger.debug("전략 B 매도: 하락 장악형 확인")

        # 모든 조건 충족
        logger.info("전략 B 매도: 모든 조건 충족!")
        return True, metadata

    def _calculate_stop_take_profit(
        self,
        signal: Signal,
        current_price: float,
        df: pd.DataFrame,
        bb_lower_val: float,
        bb_upper_val: float,
    ) -> Dict:
        """
        손절가 및 목표가 계산

        매수 시:
        - 손절: 볼린저 하단 또는 최근 20봉 저점 중 낮은 값
        - 목표: 현재가 + (리스크 * buy_risk_reward_ratio)

        매도 시:
        - 손절: 최근 20봉 고점
        - 목표: 현재가 - (리스크 * sell_risk_reward_ratio)

        Args:
            signal: 매매 신호 (BUY 또는 SELL)
            current_price: 현재가
            df: OHLCV DataFrame
            bb_lower_val: 현재 볼린저 하단 값
            bb_upper_val: 현재 볼린저 상단 값

        Returns:
            {
                "stop_loss_price": float,
                "take_profit_price": float,
                "risk_amount": float,
                "reward_amount": float,
            }
        """
        if signal == Signal.BUY:
            # 손절: 볼린저 하단 또는 최근 20봉 저점 중 낮은 값
            recent_low = df['low'].iloc[-20:].min()
            stop_loss = min(bb_lower_val, recent_low) if not pd.isna(bb_lower_val) else recent_low

            # 리스크 계산 (손절이 현재가보다 높으면 최소 1% 리스크 적용)
            risk = current_price - stop_loss
            if risk <= 0:
                risk = current_price * 0.01
                stop_loss = current_price - risk

            # 목표가
            take_profit = current_price + (risk * self._buy_risk_reward_ratio)
            reward = take_profit - current_price

            logger.debug(
                f"매수 리스크 관리: 손절={stop_loss:.2f}, 목표={take_profit:.2f}, "
                f"리스크={risk:.2f}, 리워드={reward:.2f} (비율={self._buy_risk_reward_ratio:.1f})"
            )

        else:  # SELL
            # 손절: 최근 20봉 고점
            stop_loss = df['high'].iloc[-20:].max()

            # 리스크 계산 (손절이 현재가보다 낮으면 최소 1% 리스크 적용)
            risk = stop_loss - current_price
            if risk <= 0:
                risk = current_price * 0.01
                stop_loss = current_price + risk

            # 목표가
            take_profit = current_price - (risk * self._sell_risk_reward_ratio)
            reward = current_price - take_profit

            logger.debug(
                f"매도 리스크 관리: 손절={stop_loss:.2f}, 목표={take_profit:.2f}, "
                f"리스크={risk:.2f}, 리워드={reward:.2f} (비율={self._sell_risk_reward_ratio:.1f})"
            )

        return {
            "stop_loss_price": float(stop_loss),
            "take_profit_price": float(take_profit),
            "risk_amount": float(risk),
            "reward_amount": float(reward),
        }
