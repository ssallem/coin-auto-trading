"""
리스크 관리 모듈

모든 매매 행위에 대한 안전장치를 담당한다.
주문 실행 전 리스크 검증을 수행하고, 포지션 모니터링 시 손절/익절을 판단한다.

안전장치 항목:
  1. 손절 (Stop Loss): 매수가 대비 일정 비율 이상 하락 시 자동 매도
  2. 익절 (Take Profit): 매수가 대비 일정 비율 이상 상승 시 자동 매도
  3. 트레일링 스탑: 고점 대비 일정 비율 하락 시 자동 매도
  4. 최대 투자금 제한: 총 투자 금액 상한
  5. 일일 손실 한도: 당일 누적 손실 제한
  6. 최대 동시 포지션 수 제한
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────
# 데이터 클래스 정의
# ─────────────────────────────────────

@dataclass
class Position:
    """보유 포지션 정보"""
    market: str              # 마켓 코드
    entry_price: float       # 매수 평균가
    volume: float            # 보유 수량
    entry_amount: float      # 매수 총 금액 (KRW)
    entry_time: datetime     # 매수 시각
    highest_price: float = 0.0  # 매수 이후 최고가 (트레일링 스탑용)

    def __post_init__(self):
        """최고가 초기값이 0이면 매수가로 설정한다."""
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price

    def update_highest(self, current_price: float) -> None:
        """현재가가 기존 최고가보다 높으면 최고가를 갱신한다."""
        if current_price > self.highest_price:
            self.highest_price = current_price

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """미실현 손익률(%)을 계산한다."""
        if self.entry_price == 0:
            return 0.0
        return ((current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class DailyStats:
    """일일 거래 통계"""
    date: date = field(default_factory=date.today)
    realized_pnl: float = 0.0        # 실현 손익 (KRW)
    total_trades: int = 0             # 거래 횟수
    winning_trades: int = 0           # 수익 거래 횟수
    losing_trades: int = 0            # 손실 거래 횟수

    def reset_if_new_day(self) -> None:
        """날짜가 바뀌었으면 통계를 초기화한다."""
        today = date.today()
        if self.date != today:
            logger.info(
                f"일일 통계 자동 초기화: {self.date} -> {today} "
                f"(전일 PnL: {self.realized_pnl:,.0f} KRW, "
                f"거래: {self.total_trades}건)"
            )
            self.date = today
            self.realized_pnl = 0.0
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0


class RiskCheckResult:
    """
    리스크 검증 결과

    bool()로 직접 평가할 수 있어 if 문에서 바로 사용 가능하다.
    예: if risk_manager.check_buy_allowed(amount): ...
    """

    def __init__(self, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason

    def __bool__(self) -> bool:
        return self.allowed

    def __repr__(self) -> str:
        status = "허용" if self.allowed else "차단"
        return f"RiskCheckResult({status}: {self.reason})"


# ─────────────────────────────────────
# 리스크 관리자
# ─────────────────────────────────────

class RiskManager:
    """
    리스크 관리자

    모든 주문 실행 전 리스크 검증을 수행하고,
    보유 포지션에 대해 손절/익절/트레일링 스탑을 판단한다.

    사용법:
        risk_mgr = RiskManager(
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            max_daily_loss=50000,
            max_positions=3,
            max_total_investment=1000000,
        )
        # 매수 전 검증
        check = risk_mgr.check_buy_allowed(amount=100000)
        if check:
            # 주문 실행
            ...

        # 포지션 모니터링
        action = risk_mgr.check_position("KRW-BTC", current_price=50000000)
    """

    def __init__(
        self,
        stop_loss_pct: float = 3.0,
        take_profit_pct: float = 5.0,
        max_daily_loss: float = 50_000,
        max_positions: int = 3,
        max_total_investment: float = 1_000_000,
        trailing_stop_enabled: bool = False,
        trailing_stop_pct: float = 2.0,
    ) -> None:
        """
        Args:
            stop_loss_pct: 손절 기준 비율 (%, 예: 3.0 → 3% 하락 시 손절)
            take_profit_pct: 익절 기준 비율 (%, 예: 5.0 → 5% 상승 시 익절)
            max_daily_loss: 일일 최대 허용 손실 금액 (KRW)
            max_positions: 동시 보유 가능한 최대 포지션 수
            max_total_investment: 최대 총 투자 금액 (KRW)
            trailing_stop_enabled: 트레일링 스탑 활성화 여부
            trailing_stop_pct: 트레일링 스탑 비율 (%, 고점 대비 하락 기준)
        """
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self._max_daily_loss = max_daily_loss
        self._max_positions = max_positions
        self._max_total_investment = max_total_investment
        self._trailing_stop_enabled = trailing_stop_enabled
        self._trailing_stop_pct = trailing_stop_pct

        # 보유 포지션 관리 {market: Position}
        self._positions: Dict[str, Position] = {}

        # 일일 통계
        self._daily_stats = DailyStats()

        logger.info(
            f"RiskManager 초기화: "
            f"손절={stop_loss_pct}%, 익절={take_profit_pct}%, "
            f"일일손실한도={max_daily_loss:,.0f}KRW, "
            f"최대포지션={max_positions}, "
            f"최대투자={max_total_investment:,.0f}KRW, "
            f"트레일링={'ON' if trailing_stop_enabled else 'OFF'}"
            f"({trailing_stop_pct}%)"
        )

    # ─────────────────────────────────────
    # 포지션 관리
    # ─────────────────────────────────────

    def add_position(self, position: Position) -> None:
        """
        새 포지션을 등록한다.

        Args:
            position: 등록할 Position 객체
        """
        self._positions[position.market] = position
        logger.info(
            f"포지션 등록: {position.market}, "
            f"매수가={position.entry_price:,.0f}, "
            f"수량={position.volume:.8f}, "
            f"금액={position.entry_amount:,.0f} KRW"
        )

    def remove_position(self, market: str, realized_pnl: float = 0.0) -> None:
        """
        포지션을 제거하고 일일 통계를 갱신한다.

        Args:
            market: 마켓 코드
            realized_pnl: 실현 손익 (KRW, 양수=수익, 음수=손실)
        """
        if market not in self._positions:
            logger.warning(f"포지션 제거 실패: {market} 포지션이 존재하지 않음")
            return

        del self._positions[market]

        # 일일 통계 갱신 (날짜가 바뀌었으면 자동 초기화)
        self._daily_stats.reset_if_new_day()
        self._daily_stats.realized_pnl += realized_pnl
        self._daily_stats.total_trades += 1
        if realized_pnl >= 0:
            self._daily_stats.winning_trades += 1
        else:
            self._daily_stats.losing_trades += 1

        pnl_label = "수익" if realized_pnl >= 0 else "손실"
        logger.info(
            f"포지션 제거: {market}, "
            f"실현{pnl_label}={realized_pnl:,.0f} KRW, "
            f"일일누적PnL={self._daily_stats.realized_pnl:,.0f} KRW"
        )

    def get_position(self, market: str) -> Optional[Position]:
        """
        특정 마켓의 포지션을 반환한다.

        Args:
            market: 마켓 코드
        Returns:
            Position 객체 또는 None (보유하지 않은 경우)
        """
        return self._positions.get(market)

    def get_all_positions(self) -> Dict[str, Position]:
        """
        전체 보유 포지션을 반환한다.

        Returns:
            {market: Position} 딕셔너리 (방어적 복사본)
        """
        return dict(self._positions)

    @property
    def total_invested(self) -> float:
        """현재 총 투자 금액(KRW)을 반환한다."""
        return sum(p.entry_amount for p in self._positions.values())

    @property
    def position_count(self) -> int:
        """현재 보유 포지션 수를 반환한다."""
        return len(self._positions)

    # ─────────────────────────────────────
    # 매수 리스크 검증
    # ─────────────────────────────────────

    def check_buy_allowed(self, amount: float) -> RiskCheckResult:
        """
        매수 주문이 허용되는지 검증한다.

        검증 항목 (순서대로):
          1. 최대 동시 포지션 수 초과 여부
          2. 최대 총 투자금 초과 여부
          3. 일일 손실 한도 초과 여부

        Args:
            amount: 매수 금액 (KRW)
        Returns:
            RiskCheckResult (allowed=True이면 매수 가능)
        """
        # 날짜 변경 시 일일 통계 자동 초기화
        self._daily_stats.reset_if_new_day()

        # 1. 최대 포지션 수 검증
        if self.position_count >= self._max_positions:
            reason = (
                f"최대 포지션 수 초과: "
                f"{self.position_count}/{self._max_positions}"
            )
            logger.debug(f"매수 검증 차단: {reason}")
            return RiskCheckResult(False, reason)

        # 2. 최대 투자금 검증
        if self.total_invested + amount > self._max_total_investment:
            reason = (
                f"최대 투자금 초과: "
                f"현재 {self.total_invested:,.0f} + {amount:,.0f} > "
                f"{self._max_total_investment:,.0f}"
            )
            logger.debug(f"매수 검증 차단: {reason}")
            return RiskCheckResult(False, reason)

        # 3. 일일 손실 한도 검증
        if self._daily_stats.realized_pnl < 0:
            if abs(self._daily_stats.realized_pnl) >= self._max_daily_loss:
                reason = (
                    f"일일 손실 한도 초과: "
                    f"{self._daily_stats.realized_pnl:,.0f} KRW "
                    f"(한도: -{self._max_daily_loss:,.0f})"
                )
                logger.debug(f"매수 검증 차단: {reason}")
                return RiskCheckResult(False, reason)

        logger.debug(
            f"매수 검증 통과: 금액={amount:,.0f}, "
            f"포지션={self.position_count}/{self._max_positions}, "
            f"투자={self.total_invested:,.0f}/{self._max_total_investment:,.0f}"
        )
        return RiskCheckResult(True, "매수 허용")

    # ─────────────────────────────────────
    # 포지션 모니터링 (손절/익절/트레일링)
    # ─────────────────────────────────────

    def check_position(self, market: str, current_price: float) -> Optional[str]:
        """
        보유 포지션의 리스크 상태를 확인한다.

        확인 순서:
          1. 손절: (매수가 - 현재가) / 매수가 >= stop_loss_pct
          2. 익절: (현재가 - 매수가) / 매수가 >= take_profit_pct
          3. 트레일링 스탑: 수익 구간에서 고점 대비 trailing_stop_pct 이상 하락

        Args:
            market: 마켓 코드
            current_price: 현재가
        Returns:
            "stop_loss"     - 손절 필요
            "take_profit"   - 익절 필요
            "trailing_stop" - 트레일링 스탑 발동
            None            - 유지 (조치 불필요)
        """
        position = self._positions.get(market)
        if position is None:
            return None

        # 최고가 갱신 (트레일링 스탑 추적용)
        position.update_highest(current_price)

        # 미실현 손익률 계산
        pnl_pct = position.unrealized_pnl_pct(current_price)

        # 1. 손절 검증: 손실률이 손절 기준 이상
        if pnl_pct <= -self._stop_loss_pct:
            logger.warning(
                f"[손절 발동] {market}: "
                f"손익률={pnl_pct:.2f}% (한도: -{self._stop_loss_pct}%), "
                f"매수가={position.entry_price:,.0f}, "
                f"현재가={current_price:,.0f}"
            )
            return "stop_loss"

        # 2. 익절 검증: 수익률이 익절 기준 이상
        if pnl_pct >= self._take_profit_pct:
            logger.info(
                f"[익절 발동] {market}: "
                f"손익률={pnl_pct:.2f}% (목표: +{self._take_profit_pct}%), "
                f"매수가={position.entry_price:,.0f}, "
                f"현재가={current_price:,.0f}"
            )
            return "take_profit"

        # 3. 트레일링 스탑 검증: 수익 구간에서 고점 대비 일정 비율 하락
        if self._trailing_stop_enabled and position.highest_price > 0:
            drop_from_high_pct = (
                (position.highest_price - current_price)
                / position.highest_price * 100
            )
            # 수익 구간(pnl_pct > 0)이면서, 고점 대비 하락폭이 트레일링 기준 이상
            if drop_from_high_pct >= self._trailing_stop_pct and pnl_pct > 0:
                logger.info(
                    f"[트레일링 스탑 발동] {market}: "
                    f"고점 대비 -{drop_from_high_pct:.2f}% "
                    f"(한도: -{self._trailing_stop_pct}%), "
                    f"최고가={position.highest_price:,.0f}, "
                    f"현재가={current_price:,.0f}, "
                    f"미실현 손익률={pnl_pct:.2f}%"
                )
                return "trailing_stop"

        return None

    # ─────────────────────────────────────
    # 일일 통계 및 초기화
    # ─────────────────────────────────────

    def reset_daily(self) -> None:
        """
        일일 PnL 및 통계를 수동으로 초기화한다.
        매일 자정에 스케줄러에서 호출하거나, 수동으로 호출할 수 있다.
        """
        logger.info(
            f"일일 통계 수동 초기화: "
            f"PnL={self._daily_stats.realized_pnl:,.0f} KRW, "
            f"거래={self._daily_stats.total_trades}건"
        )
        self._daily_stats = DailyStats()

    def get_daily_stats(self) -> DailyStats:
        """
        일일 거래 통계를 반환한다.
        날짜가 변경되었으면 자동으로 초기화한 뒤 반환한다.

        Returns:
            DailyStats 객체
        """
        self._daily_stats.reset_if_new_day()
        return self._daily_stats

    def get_daily_pnl(self) -> float:
        """
        당일 누적 실현 손익(KRW)을 반환한다.

        Returns:
            일일 실현 PnL (양수=수익, 음수=손실)
        """
        self._daily_stats.reset_if_new_day()
        return self._daily_stats.realized_pnl
