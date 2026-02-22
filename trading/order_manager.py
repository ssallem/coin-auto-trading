"""
주문 관리 모듈

매매 신호를 받아 실제 주문을 실행하고 관리한다.
리스크 관리자와 협력하여 주문 전 검증을 수행하고,
Notifier를 통해 매수/매도 알림을 전송한다.

주요 기능:
  - 매수/매도 주문 실행 (리스크 검증 포함)
  - 주문 이력 관리
  - Notifier 연동 매매 알림
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from api.upbit_client import OrderResult, UpbitClient
from sync.supabase_sync import SupabaseSync
from trading.risk_manager import Position, RiskManager
from utils.logger import get_logger
from utils.notifier import get_notifier

logger = get_logger(__name__)


# ─────────────────────────────────────
# 데이터 클래스 정의
# ─────────────────────────────────────

@dataclass
class TradeRecord:
    """
    거래 이력 레코드

    매수/매도 주문 실행 결과를 기록한다.
    매도 시에는 실현 손익(pnl)과 손익률(pnl_pct)이 계산된다.
    """
    market: str              # 마켓 코드
    side: str                # "buy" 또는 "sell"
    price: float             # 체결 가격
    volume: float            # 체결 수량
    amount: float            # 체결 금액 (KRW)
    timestamp: datetime      # 체결 시각
    signal_reason: str = ""  # 매매 사유 (전략 신호 or 리스크 이벤트)
    order_uuid: str = ""     # Upbit 주문 UUID
    pnl: float = 0.0        # 실현 손익 (매도 시, KRW)
    pnl_pct: float = 0.0    # 실현 손익률 (매도 시, %)


# ─────────────────────────────────────
# 주문 관리자
# ─────────────────────────────────────

class OrderManager:
    """
    주문 관리자

    매매 전략의 신호를 받아 실제 주문을 실행하고,
    RiskManager와 협력하여 안전한 거래를 보장한다.
    매수/매도 체결 시 Notifier를 통해 알림을 전송한다.

    사용법:
        order_mgr = OrderManager(client, risk_manager, per_trade_amount=100000)
        result = order_mgr.execute_buy("KRW-BTC", reason="RSI 과매도")
        result = order_mgr.execute_sell("KRW-BTC", reason="RSI 과매수")
    """

    def __init__(
        self,
        client: UpbitClient,
        risk_manager: RiskManager,
        per_trade_amount: float = 100_000,
        min_order_amount: float = 5_000,
        sync: Optional[SupabaseSync] = None,
    ) -> None:
        """
        Args:
            client: UpbitClient 인스턴스
            risk_manager: RiskManager 인스턴스
            per_trade_amount: 1회 매수 금액 (KRW)
            min_order_amount: 최소 주문 금액 (KRW, Upbit 최소 5,000원)
            sync: SupabaseSync 인스턴스 (None이면 동기화 비활성)
        """
        self._client = client
        self._risk_manager = risk_manager
        self._per_trade_amount = per_trade_amount
        self._min_order_amount = min_order_amount
        self._sync = sync

        # 거래 이력 (세션 동안 누적)
        self._trade_history: List[TradeRecord] = []

        logger.info(
            f"OrderManager 초기화: "
            f"1회매수={per_trade_amount:,.0f}KRW, "
            f"최소주문={min_order_amount:,.0f}KRW"
        )

    # ─────────────────────────────────────
    # 매수
    # ─────────────────────────────────────

    def execute_buy(
        self,
        market: str,
        amount: Optional[float] = None,
        reason: str = "",
    ) -> Optional[TradeRecord]:
        """
        시장가 매수 주문을 실행한다.

        실행 전 리스크 검증을 수행한다:
          1. 이미 해당 마켓 포지션을 보유하고 있는지 (중복 매수 차단)
          2. 최소 주문 금액 충족 여부
          3. RiskManager의 매수 허용 검증 (포지션 수, 투자금, 일일 손실)

        성공 시 RiskManager에 포지션을 등록하고, Notifier로 매수 알림을 전송한다.

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            amount: 매수 금액 (None이면 per_trade_amount 사용)
            reason: 매수 사유 (로깅/알림용)
        Returns:
            TradeRecord 또는 None (실패/차단 시)
        """
        amount = amount or self._per_trade_amount

        # 1. 이미 보유 중인지 확인 (중복 매수 차단)
        if self._risk_manager.get_position(market) is not None:
            logger.warning(f"매수 차단: {market} 이미 보유 중")
            return None

        # 2. 최소 주문 금액 확인
        if amount < self._min_order_amount:
            logger.warning(
                f"매수 차단: 금액 {amount:,.0f} < "
                f"최소 {self._min_order_amount:,.0f}"
            )
            return None

        # 3. 리스크 검증 (포지션 수, 투자금, 일일 손실 한도)
        risk_check = self._risk_manager.check_buy_allowed(amount)
        if not risk_check:
            logger.warning(f"매수 차단 (리스크): {risk_check.reason}")
            return None

        # 4. 시장가 매수 주문 실행
        try:
            order_result = self._client.buy_market_order(market, amount)
        except Exception as e:
            logger.error(f"매수 주문 API 호출 실패: {market} - {e}")
            return None

        if order_result is None:
            logger.error(f"매수 주문 실패: {market}, 금액={amount:,.0f}")
            return None

        # 5. 체결 정보 조회 (주문 UUID로 실제 체결가/체결량 확인)
        avg_price = None
        volume = None
        if order_result.uuid:
            try:
                order_detail = self._client.get_order(order_result.uuid)
                if order_detail:
                    exec_vol = order_detail.get("executed_volume")
                    if exec_vol and float(exec_vol) > 0:
                        volume = float(exec_vol)
                        # trades 내역에서 평균 체결가 계산
                        trades = order_detail.get("trades", [])
                        if trades:
                            total_funds = sum(
                                float(t.get("funds", 0)) for t in trades
                            )
                            avg_price = total_funds / volume if volume > 0 else None
            except Exception as e:
                logger.warning(f"체결 정보 조회 실패 (폴백 사용): {market} - {e}")

        # 폴백: 체결 정보 조회 실패 시 현재가 사용
        if avg_price is None or volume is None:
            fallback_price = self._client.get_current_price(market) or 0.0
            if fallback_price <= 0:
                logger.error(f"매수 후 가격 조회 실패: {market}")
                return None
            avg_price = avg_price or fallback_price
            volume = volume or (amount / fallback_price)

        # 6. RiskManager에 포지션 등록
        position = Position(
            market=market,
            entry_price=avg_price,
            volume=volume,
            entry_amount=amount,
            entry_time=datetime.now(),
        )
        self._risk_manager.add_position(position)

        # 7. 거래 이력 기록
        record = TradeRecord(
            market=market,
            side="buy",
            price=avg_price,
            volume=volume,
            amount=amount,
            timestamp=datetime.now(),
            signal_reason=reason,
            order_uuid=order_result.uuid,
        )
        self._trade_history.append(record)

        logger.info(
            f"매수 완료: {market}, "
            f"가격={avg_price:,.0f}, "
            f"수량={volume:.8f}, "
            f"금액={amount:,.0f} KRW, "
            f"사유={reason}"
        )

        # 8. Notifier로 매수 알림 전송
        self._send_trade_notification(
            market=market,
            side="buy",
            price=avg_price,
            amount=amount,
            reason=reason,
        )

        # 9. Supabase 주문 이력 동기화
        if self._sync and record:
            try:
                self._sync.push_order_history(
                    upbit_uuid=record.order_uuid,
                    market=market,
                    side="bid",
                    ord_type="price",
                    price=avg_price,
                    volume=volume,
                    amount=amount,
                    signal_reason=reason,
                    source="bot",
                )
            except Exception as e:
                logger.warning(f"매수 이력 Supabase 동기화 실패 (무시): {e}")

        return record

    # ─────────────────────────────────────
    # 매도
    # ─────────────────────────────────────

    def execute_sell(
        self,
        market: str,
        reason: str = "",
    ) -> Optional[TradeRecord]:
        """
        보유 포지션을 시장가 전량 매도한다.

        RiskManager에서 포지션을 조회하여 보유 수량을 확인하고,
        시장가 매도 주문을 실행한다. 체결 후 손익을 계산하고
        Notifier로 매도 알림을 전송한다.

        Args:
            market: 마켓 코드
            reason: 매도 사유 (로깅/알림용)
        Returns:
            TradeRecord 또는 None (실패 시)
        """
        # 1. 포지션 확인
        position = self._risk_manager.get_position(market)
        if position is None:
            logger.warning(f"매도 차단: {market} 포지션 없음")
            return None

        # 2. 보유 수량으로 시장가 매도 주문 실행
        try:
            order_result = self._client.sell_market_order(market, position.volume)
        except Exception as e:
            logger.error(f"매도 주문 API 호출 실패: {market} - {e}")
            return None

        if order_result is None:
            logger.error(f"매도 주문 실패: {market}, 수량={position.volume:.8f}")
            return None

        # 3. 체결 정보 조회 (주문 UUID로 실제 체결가/체결량 확인)
        avg_price = None
        sell_volume = None
        if order_result.uuid:
            try:
                order_detail = self._client.get_order(order_result.uuid)
                if order_detail:
                    exec_vol = order_detail.get("executed_volume")
                    if exec_vol and float(exec_vol) > 0:
                        sell_volume = float(exec_vol)
                        # trades 내역에서 평균 체결가 계산
                        trades = order_detail.get("trades", [])
                        if trades:
                            total_funds = sum(
                                float(t.get("funds", 0)) for t in trades
                            )
                            avg_price = total_funds / sell_volume if sell_volume > 0 else None
            except Exception as e:
                logger.warning(f"체결 정보 조회 실패 (폴백 사용): {market} - {e}")

        # 폴백: 체결 정보 조회 실패 시 현재가 사용
        if avg_price is None:
            avg_price = self._client.get_current_price(market) or 0.0
        if sell_volume is None:
            sell_volume = position.volume

        sell_amount = sell_volume * avg_price

        # 4. 손익 계산
        pnl = sell_amount - position.entry_amount
        pnl_pct = position.unrealized_pnl_pct(avg_price)

        # 5. RiskManager에서 포지션 제거 및 일일 통계 갱신
        self._risk_manager.remove_position(market, realized_pnl=pnl)

        # 6. 거래 이력 기록
        record = TradeRecord(
            market=market,
            side="sell",
            price=avg_price,
            volume=sell_volume,
            amount=sell_amount,
            timestamp=datetime.now(),
            signal_reason=reason,
            order_uuid=order_result.uuid,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        self._trade_history.append(record)

        pnl_label = "수익" if pnl >= 0 else "손실"
        logger.info(
            f"매도 완료: {market}, "
            f"가격={avg_price:,.0f}, "
            f"수량={sell_volume:.8f}, "
            f"금액={sell_amount:,.0f} KRW, "
            f"{pnl_label}={pnl:,.0f} KRW ({pnl_pct:+.2f}%), "
            f"사유={reason}"
        )

        # 7. Notifier로 매도 알림 전송
        self._send_trade_notification(
            market=market,
            side="sell",
            price=avg_price,
            amount=sell_amount,
            reason=reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )

        # 8. Supabase 주문 이력 동기화
        if self._sync and record:
            try:
                self._sync.push_order_history(
                    upbit_uuid=record.order_uuid,
                    market=market,
                    side="ask",
                    ord_type="market",
                    price=avg_price,
                    volume=sell_volume,
                    amount=sell_amount,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    signal_reason=reason,
                    source="bot",
                )
            except Exception as e:
                logger.warning(f"매도 이력 Supabase 동기화 실패 (무시): {e}")

        return record

    # ─────────────────────────────────────
    # 이력 조회
    # ─────────────────────────────────────

    def get_trade_history(self) -> List[TradeRecord]:
        """
        전체 거래 이력을 반환한다.

        Returns:
            TradeRecord 리스트 (시간순)
        """
        return list(self._trade_history)

    def get_trade_history_for_market(self, market: str) -> List[TradeRecord]:
        """
        특정 마켓의 거래 이력을 반환한다.

        Args:
            market: 마켓 코드
        Returns:
            해당 마켓의 TradeRecord 리스트
        """
        return [t for t in self._trade_history if t.market == market]

    def get_total_pnl(self) -> float:
        """
        전체 실현 손익 합계(KRW)를 반환한다.

        Returns:
            매도 거래들의 PnL 합계
        """
        return sum(t.pnl for t in self._trade_history if t.side == "sell")

    # ─────────────────────────────────────
    # 내부 헬퍼 - 알림 전송
    # ─────────────────────────────────────

    @staticmethod
    def _send_trade_notification(
        market: str,
        side: str,
        price: float,
        amount: float,
        reason: str = "",
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
    ) -> None:
        """
        매매 체결 알림을 Notifier를 통해 전송한다.

        Notifier가 초기화되지 않았거나 비활성화 상태이면
        아무 동작 없이 넘어간다.

        Args:
            market: 마켓 코드
            side: "buy" 또는 "sell"
            price: 체결 가격
            amount: 체결 금액
            reason: 매매 사유
            pnl: 실현 손익 (매도 시)
            pnl_pct: 실현 손익률 (매도 시)
        """
        notifier = get_notifier()
        if notifier is None:
            return

        try:
            if side == "sell" and pnl != 0.0:
                # 매도 시에는 손익 정보 포함
                pnl_label = "수익" if pnl >= 0 else "손실"
                full_reason = reason
                if full_reason:
                    full_reason += f" | {pnl_label}: {pnl:,.0f}KRW ({pnl_pct:+.2f}%)"
                else:
                    full_reason = f"{pnl_label}: {pnl:,.0f}KRW ({pnl_pct:+.2f}%)"
                notifier.send_trade_notification(
                    market=market,
                    side=side,
                    price=price,
                    amount=amount,
                    reason=full_reason,
                )
            else:
                notifier.send_trade_notification(
                    market=market,
                    side=side,
                    price=price,
                    amount=amount,
                    reason=reason,
                )
        except Exception as e:
            # 알림 전송 실패는 매매 로직에 영향을 주지 않는다
            logger.warning(f"매매 알림 전송 실패 (무시): {e}")
