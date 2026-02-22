"""
Upbit API 클라이언트

pyupbit 라이브러리를 래핑하여 일관된 인터페이스를 제공한다.
모든 API 호출은 이 클래스를 통해 이루어지며, 에러 처리와 재시도 로직을 내장한다.

주요 기능:
  - 잔고 조회 (KRW, 보유 코인)
  - 시세 조회 (현재가, 캔들 데이터, 호가)
  - 주문 실행 (시장가 매수/매도, 지정가 매수/매도)
  - 주문 조회 및 취소
  - 최대 3회 재시도 + 지수 백오프 (1초, 2초, 4초)
  - Rate Limit 준수 로깅
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import pyupbit

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────
# 데이터 클래스 정의
# ─────────────────────────────────────

@dataclass
class Balance:
    """계좌 잔고 정보"""
    currency: str          # 화폐 코드 (예: "KRW", "BTC")
    balance: float         # 보유 수량
    locked: float          # 주문 중 잠긴 수량
    avg_buy_price: float   # 평균 매수가
    unit_currency: str     # 기준 화폐 (예: "KRW")


@dataclass
class OrderResult:
    """주문 실행 결과"""
    uuid: str              # 주문 고유 번호
    side: str              # "bid" (매수) 또는 "ask" (매도)
    ord_type: str          # 주문 유형 ("price": 시장가매수, "market": 시장가매도, "limit": 지정가)
    price: Optional[float] # 주문 가격
    volume: Optional[float]  # 주문 수량
    state: str             # 주문 상태
    market: str            # 마켓 코드
    created_at: str        # 주문 생성 시각


class UpbitClient:
    """
    Upbit 거래소 API 클라이언트

    pyupbit를 래핑하여 에러 처리, 재시도, 로깅 기능을 추가한다.
    모든 거래 관련 외부 API 호출은 이 클래스를 통해 수행한다.

    재시도 정책:
      - 최대 3회 시도 (MAX_RETRIES)
      - 지수 백오프: 1초 → 2초 → 4초 (RETRY_BASE_DELAY * 2^attempt)
      - Rate Limit 초과 시에도 동일하게 재시도

    사용법:
        client = UpbitClient(access_key="...", secret_key="...")
        balances = client.get_balances()
        client.buy_market_order("KRW-BTC", 100000)
    """

    MAX_RETRIES: int = 3
    RETRY_BASE_DELAY: float = 1.0  # 기본 대기 시간 (초)

    def __init__(self, access_key: str, secret_key: str) -> None:
        """
        Upbit API 클라이언트를 초기화한다.

        Args:
            access_key: Upbit API 액세스 키
            secret_key: Upbit API 시크릿 키

        Raises:
            ValueError: API 키가 비어있는 경우
        """
        if not access_key or not secret_key:
            raise ValueError("Upbit API 키가 설정되지 않았습니다.")

        self._upbit = pyupbit.Upbit(access_key, secret_key)
        # API 호출 시각 기록 (Rate Limit 모니터링용)
        self._last_call_time: float = 0.0
        logger.info("UpbitClient 초기화 완료")

    # ─────────────────────────────────────
    # 잔고 조회
    # ─────────────────────────────────────

    def get_balances(self) -> List[Balance]:
        """
        전체 계좌 잔고를 조회한다.

        Returns:
            Balance 객체 리스트

        Raises:
            ConnectionError: 3회 재시도 후에도 API 통신 실패 시
        """
        raw = self._call_with_retry(self._upbit.get_balances)
        if raw is None:
            raise ConnectionError("잔고 조회 실패: API 응답 없음 (3회 재시도 후)")

        # pyupbit 에러 응답 처리 (dict 형태로 에러가 반환되는 경우)
        if isinstance(raw, dict) and "error" in raw:
            error_msg = raw["error"].get("message", "알 수 없는 오류")
            raise ConnectionError(f"잔고 조회 실패: {error_msg}")

        balances: List[Balance] = []
        for item in raw:
            try:
                balances.append(Balance(
                    currency=item["currency"],
                    balance=float(item["balance"]),
                    locked=float(item["locked"]),
                    avg_buy_price=float(item["avg_buy_price"]),
                    unit_currency=item["unit_currency"],
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"잔고 항목 파싱 오류 (건너뜀): {e}, 원본={item}")
                continue

        logger.debug(f"잔고 조회 완료: {len(balances)}개 항목")
        return balances

    def get_krw_balance(self) -> float:
        """
        KRW(원화) 잔고를 조회한다.

        Returns:
            KRW 잔고 (float). 보유하지 않으면 0.0
        """
        try:
            balances = self.get_balances()
            for b in balances:
                if b.currency == "KRW":
                    return b.balance
            return 0.0
        except ConnectionError as e:
            logger.error(f"KRW 잔고 조회 실패: {e}")
            return 0.0

    def get_coin_balance(self, ticker: str) -> Optional[Balance]:
        """
        특정 코인의 잔고를 조회한다.

        Args:
            ticker: 마켓 코드 (예: "KRW-BTC")

        Returns:
            Balance 객체 또는 None (보유하지 않은 경우)
        """
        # "KRW-BTC" → "BTC"
        parts = ticker.split("-")
        if len(parts) != 2:
            logger.error(f"잘못된 마켓 코드 형식: {ticker}")
            return None

        currency = parts[1]

        try:
            balances = self.get_balances()
            for b in balances:
                if b.currency == currency:
                    return b
            return None
        except ConnectionError as e:
            logger.error(f"코인 잔고 조회 실패 ({ticker}): {e}")
            return None

    # ─────────────────────────────────────
    # 시세 조회
    # ─────────────────────────────────────

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        현재가를 조회한다.

        Args:
            ticker: 마켓 코드 (예: "KRW-BTC")

        Returns:
            현재가 (float) 또는 None (실패 시)
        """
        price = self._call_with_retry(pyupbit.get_current_price, ticker)
        if price is not None:
            logger.debug(f"현재가 조회: {ticker} = {price:,.0f}")
        else:
            logger.warning(f"현재가 조회 실패: {ticker}")
        return price

    def get_ohlcv(
        self,
        ticker: str,
        interval: str = "minute15",
        count: int = 200,
    ) -> Optional[pd.DataFrame]:
        """
        OHLCV 캔들 데이터를 조회한다.

        Args:
            ticker: 마켓 코드 (예: "KRW-BTC")
            interval: 캔들 간격
                - "minute1", "minute3", "minute5", "minute10", "minute15",
                  "minute30", "minute60", "minute240"
                - "day", "week", "month"
            count: 조회 개수 (최대 200)

        Returns:
            OHLCV DataFrame (columns: open, high, low, close, volume)
            조회 실패 시 None
        """
        df = self._call_with_retry(
            pyupbit.get_ohlcv, ticker, interval=interval, count=count
        )
        if df is not None and not df.empty:
            logger.debug(f"OHLCV 조회 성공: {ticker}, interval={interval}, {len(df)}개 캔들")
        else:
            logger.warning(f"OHLCV 조회 실패 또는 빈 결과: {ticker}, interval={interval}")
        return df

    def get_orderbook(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        호가(오더북) 정보를 조회한다.

        Args:
            ticker: 마켓 코드 (예: "KRW-BTC")

        Returns:
            호가 정보 딕셔너리 또는 None (실패 시)
            - 'orderbook_units': 매수/매도 호가 리스트
            - 'total_ask_size': 매도 잔량 합계
            - 'total_bid_size': 매수 잔량 합계
        """
        orderbook = self._call_with_retry(pyupbit.get_orderbook, ticker)
        if orderbook is not None:
            logger.debug(f"호가 조회 성공: {ticker}")
        else:
            logger.warning(f"호가 조회 실패: {ticker}")
        return orderbook

    # ─────────────────────────────────────
    # 주문 실행
    # ─────────────────────────────────────

    def buy_market_order(self, ticker: str, amount: float) -> Optional[OrderResult]:
        """
        시장가 매수 주문을 실행한다.

        Args:
            ticker: 마켓 코드 (예: "KRW-BTC")
            amount: 매수 금액 (KRW). Upbit 최소 주문금액 5,000원 이상이어야 함

        Returns:
            OrderResult 또는 None (실패 시)
        """
        if amount < 5000:
            logger.error(f"시장가 매수 실패: 최소 주문금액(5,000원) 미달 - {amount:,.0f}원")
            return None

        logger.info(f"시장가 매수 주문: {ticker}, 금액={amount:,.0f} KRW")
        result = self._call_with_retry(
            self._upbit.buy_market_order, ticker, amount
        )
        order = self._parse_order_result(result)
        if order:
            logger.info(
                f"시장가 매수 주문 접수: {ticker}, uuid={order.uuid}, "
                f"금액={amount:,.0f} KRW"
            )
        return order

    def sell_market_order(self, ticker: str, volume: float) -> Optional[OrderResult]:
        """
        시장가 매도 주문을 실행한다.

        Args:
            ticker: 마켓 코드 (예: "KRW-BTC")
            volume: 매도 수량

        Returns:
            OrderResult 또는 None (실패 시)
        """
        if volume <= 0:
            logger.error(f"시장가 매도 실패: 수량이 0 이하 - {volume}")
            return None

        logger.info(f"시장가 매도 주문: {ticker}, 수량={volume}")
        result = self._call_with_retry(
            self._upbit.sell_market_order, ticker, volume
        )
        order = self._parse_order_result(result)
        if order:
            logger.info(
                f"시장가 매도 주문 접수: {ticker}, uuid={order.uuid}, "
                f"수량={volume}"
            )
        return order

    def buy_limit_order(
        self, ticker: str, price: float, volume: float
    ) -> Optional[OrderResult]:
        """
        지정가 매수 주문을 실행한다.

        Args:
            ticker: 마켓 코드
            price: 매수 희망 가격
            volume: 매수 수량

        Returns:
            OrderResult 또는 None (실패 시)
        """
        if price <= 0 or volume <= 0:
            logger.error(
                f"지정가 매수 실패: 잘못된 파라미터 - 가격={price}, 수량={volume}"
            )
            return None

        logger.info(f"지정가 매수 주문: {ticker}, 가격={price:,.0f}, 수량={volume}")
        result = self._call_with_retry(
            self._upbit.buy_limit_order, ticker, price, volume
        )
        order = self._parse_order_result(result)
        if order:
            logger.info(
                f"지정가 매수 주문 접수: {ticker}, uuid={order.uuid}, "
                f"가격={price:,.0f}, 수량={volume}"
            )
        return order

    def sell_limit_order(
        self, ticker: str, price: float, volume: float
    ) -> Optional[OrderResult]:
        """
        지정가 매도 주문을 실행한다.

        Args:
            ticker: 마켓 코드
            price: 매도 희망 가격
            volume: 매도 수량

        Returns:
            OrderResult 또는 None (실패 시)
        """
        if price <= 0 or volume <= 0:
            logger.error(
                f"지정가 매도 실패: 잘못된 파라미터 - 가격={price}, 수량={volume}"
            )
            return None

        logger.info(f"지정가 매도 주문: {ticker}, 가격={price:,.0f}, 수량={volume}")
        result = self._call_with_retry(
            self._upbit.sell_limit_order, ticker, price, volume
        )
        order = self._parse_order_result(result)
        if order:
            logger.info(
                f"지정가 매도 주문 접수: {ticker}, uuid={order.uuid}, "
                f"가격={price:,.0f}, 수량={volume}"
            )
        return order

    # ─────────────────────────────────────
    # 주문 관리
    # ─────────────────────────────────────

    def get_order(self, uuid: str) -> Optional[Dict[str, Any]]:
        """
        주문 상세 정보를 조회한다.

        Args:
            uuid: 주문 고유 번호

        Returns:
            주문 정보 딕셔너리 또는 None (실패 시)
            - uuid, side, ord_type, price, state, market,
              volume, remaining_volume, executed_volume 등
        """
        if not uuid:
            logger.error("주문 조회 실패: uuid가 비어있습니다.")
            return None

        result = self._call_with_retry(self._upbit.get_order, uuid)
        if result is not None:
            logger.debug(f"주문 조회 성공: uuid={uuid}, state={result.get('state', 'N/A')}")
        else:
            logger.warning(f"주문 조회 실패: uuid={uuid}")
        return result

    def cancel_order(self, uuid: str) -> Optional[Dict[str, Any]]:
        """
        미체결 주문을 취소한다.

        Args:
            uuid: 주문 고유 번호

        Returns:
            취소 결과 딕셔너리 또는 None (실패 시)
        """
        if not uuid:
            logger.error("주문 취소 실패: uuid가 비어있습니다.")
            return None

        logger.info(f"주문 취소 요청: uuid={uuid}")
        result = self._call_with_retry(self._upbit.cancel_order, uuid)
        if result is not None:
            logger.info(f"주문 취소 성공: uuid={uuid}")
        else:
            logger.error(f"주문 취소 실패: uuid={uuid}")
        return result

    # ─────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────

    def _call_with_retry(self, func, *args, **kwargs) -> Any:
        """
        API 호출을 재시도 로직과 함께 실행한다.

        지수 백오프 방식으로 재시도한다:
          - 1차 실패 후: 1초 대기
          - 2차 실패 후: 2초 대기
          - 3차 실패 후: 반환 (None)

        Rate Limit (429) 응답도 동일한 재시도 로직을 탄다.

        Args:
            func: 호출할 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 실행 결과 또는 None (3회 재시도 후에도 실패 시)
        """
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Rate Limit 모니터링: 호출 간격 로깅
                now = time.time()
                elapsed = now - self._last_call_time
                if self._last_call_time > 0 and elapsed < 0.1:
                    # Upbit API는 초당 10회 제한이 있으므로 100ms 미만이면 경고
                    logger.warning(
                        f"Rate Limit 주의: API 호출 간격 {elapsed:.3f}초 "
                        f"(함수: {func.__name__})"
                    )
                    # 최소 간격 보장을 위해 잠시 대기
                    sleep_time = 0.1 - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self._last_call_time = time.time()
                result = func(*args, **kwargs)

                # pyupbit가 에러를 dict로 반환하는 경우 처리
                if isinstance(result, dict) and "error" in result:
                    error_info = result["error"]
                    error_name = error_info.get("name", "unknown")
                    error_message = error_info.get("message", "알 수 없는 오류")

                    # Rate Limit 초과 (too_many_requests)
                    if error_name == "too_many_requests":
                        logger.warning(
                            f"Rate Limit 초과 (시도 {attempt}/{self.MAX_RETRIES}): "
                            f"{func.__name__} - {error_message}"
                        )
                    else:
                        logger.warning(
                            f"API 에러 응답 (시도 {attempt}/{self.MAX_RETRIES}): "
                            f"{func.__name__} - [{error_name}] {error_message}"
                        )

                    # 마지막 시도가 아니면 재시도
                    if attempt < self.MAX_RETRIES:
                        delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        logger.info(f"  → {delay:.1f}초 후 재시도...")
                        time.sleep(delay)
                        continue
                    return None

                # 정상 응답이지만 None인 경우
                if result is None:
                    logger.warning(
                        f"API 호출 결과 None (시도 {attempt}/{self.MAX_RETRIES}): "
                        f"{func.__name__}"
                    )
                    if attempt < self.MAX_RETRIES:
                        delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        logger.info(f"  → {delay:.1f}초 후 재시도...")
                        time.sleep(delay)
                        continue
                    return None

                # 성공
                return result

            except Exception as e:
                last_exception = e
                logger.error(
                    f"API 호출 예외 (시도 {attempt}/{self.MAX_RETRIES}): "
                    f"{func.__name__} - {type(e).__name__}: {e}"
                )
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.info(f"  → {delay:.1f}초 후 재시도...")
                    time.sleep(delay)

        # 모든 재시도 실패
        logger.error(
            f"API 호출 최종 실패 ({self.MAX_RETRIES}회 시도): {func.__name__}"
        )
        return None

    @staticmethod
    def _parse_order_result(raw: Any) -> Optional[OrderResult]:
        """
        API 응답을 OrderResult 데이터클래스로 변환한다.

        pyupbit 주문 응답 형식:
            {
                "uuid": "...",
                "side": "bid" | "ask",
                "ord_type": "price" | "market" | "limit",
                "price": "10000",
                "volume": "0.001",
                "state": "wait",
                "market": "KRW-BTC",
                "created_at": "2025-01-01T00:00:00+09:00",
                ...
            }

        Args:
            raw: pyupbit API 응답 (dict 또는 에러 문자열)

        Returns:
            OrderResult 객체 또는 None (파싱 실패 시)
        """
        if raw is None:
            logger.error("주문 결과 파싱 실패: 응답이 None")
            return None

        if isinstance(raw, str):
            logger.error(f"주문 실패 (문자열 응답): {raw}")
            return None

        # pyupbit 에러 응답 처리
        if isinstance(raw, dict) and "error" in raw:
            error_info = raw["error"]
            logger.error(
                f"주문 실패: [{error_info.get('name', 'unknown')}] "
                f"{error_info.get('message', '알 수 없는 오류')}"
            )
            return None

        try:
            return OrderResult(
                uuid=raw.get("uuid", ""),
                side=raw.get("side", ""),
                ord_type=raw.get("ord_type", ""),
                price=float(raw["price"]) if raw.get("price") else None,
                volume=float(raw["volume"]) if raw.get("volume") else None,
                state=raw.get("state", ""),
                market=raw.get("market", ""),
                created_at=raw.get("created_at", ""),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"주문 결과 파싱 오류: {e}, 원본 응답={raw}")
            return None
