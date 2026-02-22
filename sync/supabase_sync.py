"""
Supabase REST API 데이터 동기화 모듈

Python 봇의 계좌 스냅샷과 주문 이력을 Supabase에 동기화하고,
웹 대시보드에서 생성된 대기 주문(pending_orders)을 조회/처리한다.

환경변수:
  - SUPABASE_URL: Supabase 프로젝트 URL
  - SUPABASE_SERVICE_ROLE_KEY: 서비스 역할 키

Supabase REST API 호출 패턴은 config/supabase_loader.py를 참고한다.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

# 계좌 스냅샷 최소 간격 (초)
_SNAPSHOT_MIN_INTERVAL = 30


class SupabaseSync:
    """
    Supabase REST API를 통해 봇 데이터를 동기화한다.

    주요 기능:
      - 계좌 스냅샷 주기적 업로드 (account_snapshots)
      - 주문 이력 업로드 (order_history)
      - 대기 주문 조회 및 상태 변경 (pending_orders)

    환경변수가 설정되지 않으면 enabled=False 상태로 동작하며,
    모든 메서드는 경고 로그만 남기고 무시한다.
    """

    def __init__(self) -> None:
        self._url = os.getenv("SUPABASE_URL", "")
        self._key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self._last_snapshot_time: float = 0.0

        if self._url and self._key:
            self.enabled = True
            logger.info("SupabaseSync 초기화 완료 (활성)")
        else:
            self.enabled = False
            logger.warning(
                "SupabaseSync 비활성: SUPABASE_URL 또는 "
                "SUPABASE_SERVICE_ROLE_KEY 환경변수가 설정되지 않았습니다."
            )

    # ─────────────────────────────────────
    # 공통 헬퍼
    # ─────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        """Supabase REST API 공통 요청 헤더를 반환한다."""
        return {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

    # ─────────────────────────────────────
    # 계좌 스냅샷
    # ─────────────────────────────────────

    def push_account_snapshot(
        self,
        krw_balance: float,
        krw_locked: float,
        holdings: List[Dict[str, Any]],
        force: bool = False,
    ) -> None:
        """
        account_snapshots 테이블에 현재 계좌 상태를 INSERT 한다.

        force=False 이면 마지막 스냅샷으로부터 최소 30초 경과 후에만 전송한다.

        Args:
            krw_balance: KRW 가용 잔고
            krw_locked: KRW 주문 대기 잠금 금액
            holdings: 보유 코인 리스트
                [{"currency": "BTC", "balance": 0.001, "locked": 0.0,
                  "avg_buy_price": 50000000}]
            force: True면 주기 무시하고 즉시 전송
        """
        if not self.enabled:
            return

        now = time.time()
        if not force and (now - self._last_snapshot_time) < _SNAPSHOT_MIN_INTERVAL:
            return

        payload = {
            "bot_id": "main",
            "krw_balance": krw_balance,
            "krw_locked": krw_locked,
            "holdings": holdings,
            "snapshot_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            resp = requests.post(
                f"{self._url}/rest/v1/account_snapshots",
                headers=self._headers(),
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            self._last_snapshot_time = time.time()
            logger.debug(
                f"계좌 스냅샷 전송 완료: KRW={krw_balance:,.0f}, "
                f"보유코인={len(holdings)}종"
            )
        except Exception as e:
            logger.warning(f"계좌 스냅샷 전송 실패 (무시): {e}")

    # ─────────────────────────────────────
    # 주문 이력
    # ─────────────────────────────────────

    def push_order_history(
        self,
        upbit_uuid: str,
        market: str,
        side: str,
        ord_type: str,
        price: float,
        volume: float,
        amount: float,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        signal_reason: str = "",
        source: str = "bot",
        pending_order_id: Optional[int] = None,
    ) -> None:
        """
        order_history 테이블에 체결 내역을 INSERT 한다.

        upbit_uuid 컬럼의 UNIQUE 제약으로 중복 삽입을 방지한다.

        Args:
            upbit_uuid: Upbit 주문 UUID
            market: 마켓 코드 (예: "KRW-BTC")
            side: "bid" (매수) 또는 "ask" (매도)
            ord_type: 주문 유형 ("price", "market", "limit")
            price: 체결 가격
            volume: 체결 수량
            amount: 체결 금액 (KRW)
            pnl: 실현 손익 (KRW, 매도 시)
            pnl_pct: 실현 손익률 (%, 매도 시)
            signal_reason: 매매 사유
            source: 주문 출처 ("bot" 또는 "web")
            pending_order_id: 웹 대기 주문 ID (있는 경우)
        """
        if not self.enabled:
            return

        payload: Dict[str, Any] = {
            "bot_id": "main",
            "upbit_uuid": upbit_uuid,
            "market": market,
            "side": side,
            "ord_type": ord_type,
            "price": price,
            "volume": volume,
            "amount": amount,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "signal_reason": signal_reason,
            "source": source,
        }
        if pending_order_id is not None:
            payload["pending_order_id"] = pending_order_id

        try:
            resp = requests.post(
                f"{self._url}/rest/v1/order_history",
                headers=self._headers(),
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            logger.debug(
                f"주문 이력 전송 완료: {market} {side} "
                f"uuid={upbit_uuid[:8]}..."
            )
        except Exception as e:
            logger.warning(f"주문 이력 전송 실패 (무시): {e}")

    # ─────────────────────────────────────
    # 대기 주문 (pending_orders)
    # ─────────────────────────────────────

    def fetch_pending_orders(self) -> List[Dict[str, Any]]:
        """
        status='pending' 인 대기 주문을 오래된 순으로 최대 10건 조회한다.

        Returns:
            대기 주문 딕셔너리 리스트. 비활성 상태이거나 오류 시 빈 리스트.
        """
        if not self.enabled:
            return []

        try:
            resp = requests.get(
                f"{self._url}/rest/v1/pending_orders",
                headers={
                    "apikey": self._key,
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                params={
                    "bot_id": "eq.main",
                    "status": "eq.pending",
                    "select": "*",
                    "order": "requested_at.asc",
                    "limit": "10",
                },
                timeout=10,
            )
            resp.raise_for_status()
            orders = resp.json()
            if orders:
                logger.debug(f"대기 주문 {len(orders)}건 조회됨")
            return orders
        except Exception as e:
            logger.warning(f"대기 주문 조회 실패 (무시): {e}")
            return []

    def mark_pending_processing(self, order_id: int) -> None:
        """대기 주문 상태를 'processing'으로 변경한다."""
        self._update_pending_status(order_id, "processing", current_status="pending")

    def mark_pending_done(self, order_id: int, upbit_uuid: str) -> None:
        """대기 주문 상태를 'done'으로 변경하고 Upbit UUID를 기록한다."""
        self._update_pending_status(
            order_id, "done", extra={"upbit_uuid": upbit_uuid}
        )

    def mark_pending_failed(self, order_id: int, error: str) -> None:
        """대기 주문 상태를 'failed'로 변경하고 에러 메시지를 기록한다."""
        self._update_pending_status(
            order_id, "failed", extra={"error_message": error}
        )

    def _update_pending_status(
        self,
        order_id: int,
        status: str,
        extra: Optional[Dict[str, Any]] = None,
        current_status: Optional[str] = None,
    ) -> None:
        """
        pending_orders 테이블의 상태를 변경하는 공통 구현.

        Args:
            order_id: 대기 주문 ID (PK)
            status: 변경할 상태 ("processing", "done", "failed")
            extra: 추가로 업데이트할 컬럼 딕셔너리
            current_status: 낙관적 잠금을 위한 현재 상태 조건 (지정 시 해당 상태인 행만 업데이트)
        """
        if not self.enabled:
            return

        payload: Dict[str, Any] = {"status": status}
        if extra:
            payload.update(extra)

        params: Dict[str, str] = {"id": f"eq.{order_id}"}
        if current_status:
            params["status"] = f"eq.{current_status}"

        try:
            resp = requests.patch(
                f"{self._url}/rest/v1/pending_orders",
                headers=self._headers(),
                params=params,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            logger.debug(
                f"대기 주문 #{order_id} 상태 변경: {status}"
            )
        except Exception as e:
            logger.warning(
                f"대기 주문 #{order_id} 상태 변경 실패 (무시): {e}"
            )
