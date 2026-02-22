"""
백테스트 엔진

과거 데이터를 이용하여 매매 전략의 성과를 검증한다.
실제 거래 없이 전략의 수익률, 승률, MDD 등을 시뮬레이션한다.

주요 기능:
  - 과거 캔들 데이터 기반 전략 시뮬레이션
  - 수수료 반영
  - 성과 지표 계산 (총 수익률, 승률, MDD, 샤프비율, Profit Factor 등)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from api.upbit_client import UpbitClient
from data.indicators import Indicators
from strategies.base_strategy import BaseStrategy, Signal
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """백테스트 개별 거래 기록"""
    market: str
    side: str                  # "buy" 또는 "sell"
    price: float               # 체결 가격
    amount: float              # 체결 금액 (KRW)
    timestamp: datetime        # 체결 시각
    reason: str = ""           # 매매 사유
    pnl: float = 0.0           # 실현 손익 (매도 시, 수수료 차감 후)
    pnl_pct: float = 0.0       # 수익률 % (매도 시)


@dataclass
class BacktestResult:
    """백테스트 결과 요약"""
    # 기본 정보
    initial_capital: float
    final_capital: float

    # 성과 지표
    total_return_pct: float       # 총 수익률 (%)
    win_rate: float               # 승률 (%) - 수익 거래 / 전체 매도 거래
    total_trades: int             # 총 매매 횟수 (매수 + 매도)
    win_count: int                # 수익 거래 수
    loss_count: int               # 손실 거래 수
    max_drawdown_pct: float       # 최대 낙폭 MDD (%)
    sharpe_ratio: float           # 샤프 비율 (일일 수익률 기반, 연율화)
    profit_factor: float          # 총이익 / 총손실

    # 상세 거래 기록
    trades: List[BacktestTrade] = field(default_factory=list)

    # 잔고 시계열 (시각화용)
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """결과 요약 문자열을 반환한다."""
        sell_trades = self.win_count + self.loss_count
        return (
            f"\n{'=' * 55}\n"
            f"  백테스트 결과 요약\n"
            f"{'=' * 55}\n"
            f"  초기 자본:       {self.initial_capital:>15,.0f} KRW\n"
            f"  최종 자본:       {self.final_capital:>15,.0f} KRW\n"
            f"  총 수익률:       {self.total_return_pct:>+14.2f} %\n"
            f"{'─' * 55}\n"
            f"  총 거래 횟수:    {self.total_trades:>15d} 건\n"
            f"  매도 거래:       {sell_trades:>15d} 건\n"
            f"    - 수익 거래:   {self.win_count:>15d} 건\n"
            f"    - 손실 거래:   {self.loss_count:>15d} 건\n"
            f"  승률:            {self.win_rate:>14.1f} %\n"
            f"{'─' * 55}\n"
            f"  최대 낙폭(MDD):  {self.max_drawdown_pct:>14.2f} %\n"
            f"  샤프 비율:       {self.sharpe_ratio:>14.2f}\n"
            f"  Profit Factor:   {self.profit_factor:>14.2f}\n"
            f"{'=' * 55}"
        )


class BacktestEngine:
    """
    백테스트 엔진

    과거 캔들 데이터를 순차적으로 전략에 입력하여
    매매 시뮬레이션을 수행하고 성과를 분석한다.

    사용법:
        engine = BacktestEngine(
            client=client,
            strategy=RSIStrategy(period=14, oversold=30, overbought=70),
            initial_capital=1_000_000,
            commission_rate=0.0005,
        )
        result = engine.run("KRW-BTC", interval="day", count=200)
        print(result.summary())
    """

    def __init__(
        self,
        client: UpbitClient,
        strategy: BaseStrategy,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.0005,
        per_trade_amount: Optional[float] = None,
    ) -> None:
        """
        Args:
            client: UpbitClient (과거 데이터 조회용)
            strategy: 테스트할 전략 인스턴스
            initial_capital: 초기 자본금 (KRW)
            commission_rate: 수수료율 (0.0005 = 0.05 %)
            per_trade_amount: 1회 매수 금액 (None이면 전체 자본금 사용)
        """
        self._client = client
        self._strategy = strategy
        self._initial_capital = initial_capital
        self._commission_rate = commission_rate
        self._per_trade_amount = per_trade_amount

    # ─────────────────────────────────────
    # 전략별 최소 워밍업 봉 수 결정
    # ─────────────────────────────────────

    @staticmethod
    def _get_warmup_bars(strategy: BaseStrategy) -> int:
        """
        전략이 유효한 신호를 내기 위해 필요한 최소 봉 수를 반환한다.
        전략 유형에 따라 동적으로 결정한다.
        지표 계산이 안정적으로 이루어지려면 추가 마진이 필요하므로
        약간의 여유를 더한다.
        """
        # RSI 전략
        if hasattr(strategy, "_period") and strategy.name == "rsi":
            return strategy._period + 5

        # MA Cross 전략: 장기 이동평균 기간 + 여유
        if hasattr(strategy, "_long_period") and strategy.name == "ma_cross":
            return strategy._long_period + 5

        # 볼린저 밴드 전략
        if hasattr(strategy, "_period") and strategy.name == "bollinger":
            return strategy._period + 5

        # 기본 워밍업 (MACD 26+9=35봉 등을 고려해 50으로 설정)
        return 50

    # ─────────────────────────────────────
    # 메인 실행
    # ─────────────────────────────────────

    def run(
        self,
        market: str,
        interval: str = "day",
        count: int = 200,
        df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        백테스트를 실행한다.

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            interval: 캔들 간격 (예: "day", "minute60")
            count: 캔들 개수
            df: 외부에서 제공한 OHLCV DataFrame (None이면 API로 조회)
        Returns:
            BacktestResult 객체
        Raises:
            ValueError: 데이터를 가져올 수 없는 경우
        """
        # ── 1. 데이터 준비 ──
        if df is None:
            df = self._client.get_ohlcv(market, interval=interval, count=count)
        if df is None or df.empty:
            raise ValueError(f"백테스트 데이터를 가져올 수 없습니다: {market}")

        # ── 2. 지표 일괄 추가 ──
        df = Indicators.add_all_indicators(df)
        if df.empty:
            raise ValueError(f"지표 계산에 실패했습니다: {market}")

        logger.info(
            "백테스트 시작: %s / %s, 데이터: %d개 캔들, 초기자본: %s KRW",
            self._strategy.name,
            market,
            len(df),
            f"{self._initial_capital:,.0f}",
        )

        # ── 3. 전략 상태 초기화 ──
        self._strategy.reset()

        # ── 4. 시뮬레이션 변수 초기화 ──
        capital = self._initial_capital        # 현재 현금
        position_price: float = 0.0            # 매수 평균가
        position_volume: float = 0.0           # 보유 수량
        in_position: bool = False
        entry_reason: str = ""

        trades: List[BacktestTrade] = []
        equity_curve: List[float] = [self._initial_capital]

        # 워밍업 인덱스: 지표가 안정적으로 계산된 이후부터 거래 시작
        warmup = self._get_warmup_bars(self._strategy)
        start_idx = min(warmup, len(df) - 1)

        logger.debug(
            "워밍업 구간: %d봉 (인덱스 %d부터 시뮬레이션 시작)", warmup, start_idx
        )

        # ── 5. 시뮬레이션 루프 ──
        for i in range(start_idx, len(df)):
            # 현재 봉까지의 히스토리컬 데이터 슬라이스
            historical = df.iloc[: i + 1]
            current_price = float(df.iloc[i]["close"])

            # 타임스탬프 추출 (DatetimeIndex가 아닐 경우 대비)
            idx_val = df.index[i]
            current_time = (
                idx_val
                if hasattr(idx_val, "strftime")
                else datetime.now()
            )

            # 전략 신호 분석
            signal_result = self._strategy.analyze(market, historical, current_price)

            # ── 매수 처리 ──
            if signal_result.signal == Signal.BUY and not in_position:
                # per_trade_amount가 설정되어 있으면 해당 금액만, 아니면 전체 자본 투입
                if self._per_trade_amount is not None:
                    buy_amount = min(self._per_trade_amount, capital)
                else:
                    buy_amount = capital
                if buy_amount <= 5000:
                    # 최소 주문 금액 미달 → 스킵
                    equity_curve.append(capital)
                    continue

                commission = buy_amount * self._commission_rate
                net_invest = buy_amount - commission  # 실제 코인 매수에 사용되는 금액
                position_volume = net_invest / current_price
                position_price = current_price
                capital -= buy_amount  # 매수 금액만큼 현금 차감
                in_position = True
                entry_reason = signal_result.reason

                trades.append(BacktestTrade(
                    market=market,
                    side="buy",
                    price=current_price,
                    amount=buy_amount,
                    timestamp=current_time,
                    reason=signal_result.reason,
                ))
                logger.debug(
                    "[%d] 매수: price=%.2f, volume=%.8f, 투입=%.0f KRW",
                    i, current_price, position_volume, buy_amount,
                )

            # ── 매도 처리 ──
            elif signal_result.signal == Signal.SELL and in_position:
                sell_gross = position_volume * current_price
                commission = sell_gross * self._commission_rate
                sell_net = sell_gross - commission

                # 손익 계산 (매수 시 투입한 순수 금액 대비)
                cost_basis = position_volume * position_price
                pnl = sell_net - cost_basis
                pnl_pct = (
                    (current_price - position_price) / position_price * 100
                    if position_price > 0
                    else 0.0
                )

                capital += sell_net
                in_position = False

                trades.append(BacktestTrade(
                    market=market,
                    side="sell",
                    price=current_price,
                    amount=sell_gross,
                    timestamp=current_time,
                    reason=signal_result.reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                ))
                logger.debug(
                    "[%d] 매도: price=%.2f, pnl=%+.0f (%.2f%%)",
                    i, current_price, pnl, pnl_pct,
                )

                position_price = 0.0
                position_volume = 0.0

            # ── 자산 가치 기록 (현금 + 보유 코인 평가액) ──
            current_equity = capital
            if in_position:
                current_equity += position_volume * current_price
            equity_curve.append(current_equity)

        # ── 6. 미청산 포지션 강제 청산 ──
        if in_position:
            final_price = float(df.iloc[-1]["close"])
            sell_gross = position_volume * final_price
            commission = sell_gross * self._commission_rate
            sell_net = sell_gross - commission

            cost_basis = position_volume * position_price
            pnl = sell_net - cost_basis
            pnl_pct = (
                (final_price - position_price) / position_price * 100
                if position_price > 0
                else 0.0
            )

            capital += sell_net

            final_time = df.index[-1]
            if not hasattr(final_time, "strftime"):
                final_time = datetime.now()

            trades.append(BacktestTrade(
                market=market,
                side="sell",
                price=final_price,
                amount=sell_gross,
                timestamp=final_time,
                reason="백테스트 종료 (강제 청산)",
                pnl=pnl,
                pnl_pct=pnl_pct,
            ))
            logger.info(
                "미청산 포지션 강제 청산: price=%.2f, pnl=%+.0f (%.2f%%)",
                final_price, pnl, pnl_pct,
            )

            # equity_curve 마지막 값 보정
            if equity_curve:
                equity_curve[-1] = capital

        # ── 7. 성과 지표 계산 ──
        result = self._calculate_metrics(
            trades=trades,
            final_capital=capital,
            equity_curve=equity_curve,
        )

        logger.info(result.summary())
        return result

    # ─────────────────────────────────────
    # 성과 지표 계산
    # ─────────────────────────────────────

    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        final_capital: float,
        equity_curve: List[float],
    ) -> BacktestResult:
        """매매 기록과 equity curve로부터 성과 지표를 산출한다."""

        # 매도 거래만 추출 (수익/손실 판정 대상)
        sell_trades = [t for t in trades if t.side == "sell"]
        winning = [t for t in sell_trades if t.pnl > 0]
        losing = [t for t in sell_trades if t.pnl <= 0]

        total_trades = len(trades)
        sell_count = len(sell_trades)
        win_count = len(winning)
        loss_count = len(losing)

        # 총 수익률
        total_return_pct = (
            (final_capital - self._initial_capital) / self._initial_capital * 100
        )

        # 승률
        win_rate = (win_count / sell_count * 100) if sell_count > 0 else 0.0

        # MDD 계산
        max_drawdown_pct = self._calculate_mdd(equity_curve)

        # 샤프 비율 (일일 수익률 기반, 연율화)
        sharpe_ratio = self._calculate_sharpe(equity_curve)

        # Profit Factor = 총이익 / 총손실 (절대값)
        total_profit = sum(t.pnl for t in winning)
        total_loss = abs(sum(t.pnl for t in losing))
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        elif total_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        return BacktestResult(
            initial_capital=self._initial_capital,
            final_capital=round(final_capital, 2),
            total_return_pct=round(total_return_pct, 2),
            win_rate=round(win_rate, 1),
            total_trades=total_trades,
            win_count=win_count,
            loss_count=loss_count,
            max_drawdown_pct=round(max_drawdown_pct, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            profit_factor=round(profit_factor, 2),
            trades=trades,
            equity_curve=equity_curve,
        )

    # ─────────────────────────────────────
    # MDD (최대 낙폭)
    # ─────────────────────────────────────

    @staticmethod
    def _calculate_mdd(equity_curve: List[float]) -> float:
        """
        최대 낙폭(Maximum Drawdown)을 계산한다.

        equity_curve의 고점 대비 하락 비율의 최대값을 반환한다.

        Returns:
            MDD (%) - 항상 양수 또는 0
        """
        if len(equity_curve) < 2:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        return max_dd

    # ─────────────────────────────────────
    # 샤프 비율 (연율화)
    # ─────────────────────────────────────

    @staticmethod
    def _calculate_sharpe(equity_curve: List[float]) -> float:
        """
        일일 수익률 기반 샤프 비율을 계산한다.

        Sharpe Ratio = (평균 일일 수익률 / 일일 수익률 표준편차) * sqrt(252)

        무위험 이자율은 0으로 가정한다.
        수익률 데이터가 2개 미만이거나 표준편차가 0이면 0.0을 반환한다.

        Returns:
            샤프 비율 (연율화)
        """
        if len(equity_curve) < 3:
            return 0.0

        # equity_curve를 numpy 배열로 변환
        eq = np.array(equity_curve, dtype=float)

        # 연속 수익률(일일 수익률) 계산: r_t = (E_t - E_{t-1}) / E_{t-1}
        # 0인 구간을 방어하기 위해 이전 값이 0인 경우 건너뜀
        prev = eq[:-1]
        curr = eq[1:]

        # 0 나눗셈 방지: 이전 값이 0이 아닌 인덱스만 선택
        valid_mask = prev > 0
        if valid_mask.sum() < 2:
            return 0.0

        daily_returns = (curr[valid_mask] - prev[valid_mask]) / prev[valid_mask]

        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns, ddof=1)  # 표본 표준편차

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        # 연율화: sqrt(252)
        sharpe = (avg_return / std_return) * math.sqrt(252)
        return sharpe
