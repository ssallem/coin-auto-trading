"""
데이터 수집기

Upbit API와 WebSocket을 통해 시세 데이터를 수집하고 관리한다.
REST API로 캔들 데이터를, WebSocket으로 실시간 체결 데이터를 수집한다.

주요 기능:
  - 캔들(OHLCV) 데이터 수집 및 TTL 기반 캐싱
  - WebSocket을 통한 실시간 시세 스트리밍
  - 데이터 캐싱 및 자동 갱신 관리
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import websockets

from api.upbit_client import UpbitClient
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────
# 데이터 클래스 정의
# ─────────────────────────────────────

@dataclass
class TickerSnapshot:
    """특정 마켓의 실시간 시세 스냅샷"""
    market: str              # 마켓 코드 (예: "KRW-BTC")
    current_price: float     # 현재가
    high_price: float        # 당일 고가
    low_price: float         # 당일 저가
    trade_volume: float      # 24시간 누적 거래량
    change_rate: float       # 전일 대비 변화율 (부호 포함)
    timestamp: datetime      # 수신 시각


@dataclass
class _CacheEntry:
    """캔들 데이터 캐시 항목 (내부용)"""
    data: pd.DataFrame       # 캔들 데이터
    timestamp: float         # 캐시 저장 시각 (time.time())
    interval: str            # 캔들 간격
    count: int               # 캔들 개수


class DataCollector:
    """
    시세 데이터 수집 및 관리

    두 가지 모드를 제공한다:
      1. 폴링 모드: REST API를 호출하여 캔들 데이터 수집 (TTL 기반 캐싱)
      2. 스트리밍 모드: WebSocket으로 실시간 체결가 수신

    캔들 데이터는 마켓별로 캐싱되며, TTL(기본 60초)이 만료되면 자동 갱신된다.

    사용법:
        collector = DataCollector(
            client=upbit_client,
            markets=["KRW-BTC", "KRW-ETH"],
            timeframe="minute15",
            candle_count=200,
        )

        # REST: 캔들 데이터 조회 (TTL 캐싱 적용)
        df = collector.get_candles("KRW-BTC")

        # REST: 전체 마켓 캔들 갱신
        all_data = collector.refresh_all_candles()

        # WebSocket: 실시간 시세 수신
        collector.start_websocket(on_tick=my_callback)
        snapshot = collector.get_snapshot("KRW-BTC")
        collector.stop_websocket()
    """

    UPBIT_WS_URL: str = "wss://api.upbit.com/websocket/v1"
    DEFAULT_CACHE_TTL: float = 60.0  # 캐시 유효 기간 (초)

    def __init__(
        self,
        client: UpbitClient,
        markets: List[str],
        timeframe: str = "minute15",
        candle_count: int = 200,
        cache_ttl: float = 60.0,
    ) -> None:
        """
        데이터 수집기를 초기화한다.

        Args:
            client: UpbitClient 인스턴스
            markets: 감시 대상 마켓 코드 리스트 (예: ["KRW-BTC", "KRW-ETH"])
            timeframe: 캔들 기본 시간프레임 (예: "minute15", "day")
            candle_count: 캔들 기본 조회 개수 (최대 200)
            cache_ttl: 캐시 유효 기간 (초, 기본 60초)
        """
        self._client = client
        self._markets = list(markets)  # 방어적 복사
        self._timeframe = timeframe
        self._candle_count = candle_count
        self._cache_ttl = cache_ttl

        # 마켓별 캔들 데이터 캐시 {market: _CacheEntry}
        self._candle_cache: Dict[str, _CacheEntry] = {}
        # 캐시 접근 동기화용 락
        self._cache_lock = threading.Lock()

        # 마켓별 실시간 스냅샷 {market: TickerSnapshot}
        self._snapshots: Dict[str, TickerSnapshot] = {}
        # 스냅샷 접근 동기화용 락
        self._snapshot_lock = threading.Lock()

        # WebSocket 스레드 관리
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_running = False
        self._tick_callbacks: List[Callable[[TickerSnapshot], None]] = []

        logger.info(
            f"DataCollector 초기화 완료: markets={self._markets}, "
            f"timeframe={self._timeframe}, candle_count={self._candle_count}, "
            f"cache_ttl={self._cache_ttl}초"
        )

    @property
    def markets(self) -> List[str]:
        """감시 대상 마켓 코드 리스트를 반환한다."""
        return list(self._markets)

    # ─────────────────────────────────────
    # 캔들 데이터 (REST API)
    # ─────────────────────────────────────

    def get_candles(
        self,
        market: str,
        interval: Optional[str] = None,
        count: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        캔들(OHLCV) 데이터를 조회한다.

        TTL 기반 캐싱을 적용한다:
          - use_cache=True이고 캐시가 유효하면(TTL 미만) 캐시를 반환
          - 캐시가 없거나 TTL이 만료되었으면 API를 호출하고 캐시를 갱신

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            interval: 캔들 간격 (None이면 생성자의 기본값 사용)
            count: 조회 개수 (None이면 생성자의 기본값 사용)
            use_cache: 캐시 사용 여부 (False이면 항상 API 호출)

        Returns:
            OHLCV DataFrame (columns: open, high, low, close, volume) 또는 None
        """
        interval = interval or self._timeframe
        count = count or self._candle_count

        # 캐시 확인
        if use_cache:
            with self._cache_lock:
                cached = self._candle_cache.get(market)
                if cached is not None:
                    age = time.time() - cached.timestamp
                    # TTL 유효 + 동일 interval + 충분한 데이터 수
                    if (
                        age < self._cache_ttl
                        and cached.interval == interval
                        and len(cached.data) >= count
                    ):
                        logger.debug(
                            f"캐시 히트: {market}, "
                            f"잔여 TTL={self._cache_ttl - age:.1f}초, "
                            f"{len(cached.data)}개 캔들"
                        )
                        return cached.data.copy()
                    else:
                        logger.debug(
                            f"캐시 만료/불일치: {market}, "
                            f"age={age:.1f}초, interval={cached.interval}"
                        )

        # API 호출
        df = self._client.get_ohlcv(market, interval=interval, count=count)
        if df is not None and not df.empty:
            with self._cache_lock:
                self._candle_cache[market] = _CacheEntry(
                    data=df.copy(),
                    timestamp=time.time(),
                    interval=interval,
                    count=len(df),
                )
            logger.info(f"캔들 데이터 갱신: {market}, interval={interval}, {len(df)}개")
        else:
            logger.warning(f"캔들 데이터 조회 실패: {market}, interval={interval}")

        return df

    def refresh_all_candles(self) -> Dict[str, pd.DataFrame]:
        """
        모든 감시 대상 마켓의 캔들 데이터를 강제 갱신한다.

        캐시를 무시하고 모든 마켓에 대해 API를 호출한다.
        Upbit Rate Limit을 고려하여 각 호출 사이에 0.1초 간격을 둔다.

        Returns:
            {market: DataFrame} 딕셔너리 (조회 실패한 마켓은 제외)
        """
        result: Dict[str, pd.DataFrame] = {}
        logger.info(f"전체 캔들 갱신 시작: {len(self._markets)}개 마켓")

        for i, market in enumerate(self._markets):
            df = self.get_candles(market, use_cache=False)
            if df is not None:
                result[market] = df

            # Rate Limit 보호: 마지막 마켓이 아니면 잠시 대기
            if i < len(self._markets) - 1:
                time.sleep(0.1)

        logger.info(
            f"전체 캔들 갱신 완료: "
            f"성공={len(result)}/{len(self._markets)}개 마켓"
        )
        return result

    def invalidate_cache(self, market: Optional[str] = None) -> None:
        """
        캐시를 무효화한다.

        Args:
            market: 특정 마켓만 무효화 (None이면 전체 캐시 삭제)
        """
        with self._cache_lock:
            if market:
                self._candle_cache.pop(market, None)
                logger.debug(f"캐시 무효화: {market}")
            else:
                self._candle_cache.clear()
                logger.debug("전체 캐시 무효화")

    # ─────────────────────────────────────
    # 실시간 시세 (WebSocket)
    # ─────────────────────────────────────

    def start_websocket(
        self,
        on_tick: Optional[Callable[[TickerSnapshot], None]] = None,
    ) -> None:
        """
        WebSocket 연결을 시작하여 실시간 체결가를 수신한다.
        별도 데몬 스레드에서 비동기로 실행된다.

        Args:
            on_tick: 체결 데이터 수신 시 호출되는 콜백 함수.
                     TickerSnapshot 객체를 인자로 받는다.
        """
        if self._ws_running:
            logger.warning("WebSocket이 이미 실행 중입니다.")
            return

        if not self._markets:
            logger.error("WebSocket 시작 실패: 감시 대상 마켓이 없습니다.")
            return

        if on_tick is not None:
            self._tick_callbacks.append(on_tick)

        self._ws_running = True
        self._ws_thread = threading.Thread(
            target=self._run_websocket_loop,
            daemon=True,
            name="ws-data-collector",
        )
        self._ws_thread.start()
        logger.info(f"WebSocket 시작: markets={self._markets}")

    def stop_websocket(self) -> None:
        """
        WebSocket 연결을 종료한다.
        스레드가 정상 종료될 때까지 최대 5초 대기한다.
        """
        if not self._ws_running:
            logger.debug("WebSocket이 이미 종료된 상태입니다.")
            return

        logger.info("WebSocket 종료 요청...")
        self._ws_running = False

        if self._ws_thread is not None and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5)
            if self._ws_thread.is_alive():
                logger.warning("WebSocket 스레드가 5초 내에 종료되지 않았습니다.")
            else:
                logger.info("WebSocket 종료 완료")

        self._ws_thread = None

    def get_snapshot(self, market: str) -> Optional[TickerSnapshot]:
        """
        특정 마켓의 최신 실시간 스냅샷을 반환한다.

        WebSocket이 실행 중이어야 스냅샷 데이터가 갱신된다.

        Args:
            market: 마켓 코드 (예: "KRW-BTC")

        Returns:
            TickerSnapshot 또는 None (아직 수신된 데이터가 없는 경우)
        """
        with self._snapshot_lock:
            return self._snapshots.get(market)

    def get_all_snapshots(self) -> Dict[str, TickerSnapshot]:
        """
        모든 마켓의 최신 실시간 스냅샷을 반환한다.

        Returns:
            {market: TickerSnapshot} 딕셔너리
        """
        with self._snapshot_lock:
            return dict(self._snapshots)

    def add_tick_callback(self, callback: Callable[[TickerSnapshot], None]) -> None:
        """
        실시간 체결가 수신 콜백을 등록한다.

        WebSocket 실행 중에도 동적으로 콜백을 추가할 수 있다.

        Args:
            callback: TickerSnapshot 객체를 인자로 받는 콜백 함수
        """
        self._tick_callbacks.append(callback)
        logger.debug(f"틱 콜백 등록: 총 {len(self._tick_callbacks)}개")

    def remove_tick_callback(self, callback: Callable[[TickerSnapshot], None]) -> None:
        """
        등록된 틱 콜백을 제거한다.

        Args:
            callback: 제거할 콜백 함수
        """
        try:
            self._tick_callbacks.remove(callback)
            logger.debug(f"틱 콜백 제거: 남은 {len(self._tick_callbacks)}개")
        except ValueError:
            logger.warning("제거하려는 콜백이 등록되어 있지 않습니다.")

    @property
    def is_websocket_running(self) -> bool:
        """WebSocket이 현재 실행 중인지 반환한다."""
        return self._ws_running

    # ─────────────────────────────────────
    # 내부 헬퍼 - WebSocket
    # ─────────────────────────────────────

    def _run_websocket_loop(self) -> None:
        """
        WebSocket 이벤트 루프를 실행한다 (별도 스레드).

        새로운 asyncio 이벤트 루프를 생성하여 _websocket_handler를
        코루틴으로 실행한다. 스레드 종료 시 이벤트 루프를 정리한다.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._websocket_handler())
        except Exception as e:
            logger.error(f"WebSocket 이벤트 루프 오류: {type(e).__name__}: {e}")
        finally:
            # 루프 정리: 모든 pending 태스크 취소
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            finally:
                loop.close()
                logger.debug("WebSocket 이벤트 루프 종료")

    async def _websocket_handler(self) -> None:
        """
        WebSocket 연결 및 메시지 수신 핸들러.

        Upbit WebSocket API에 연결하여 실시간 ticker 데이터를 구독한다.
        연결이 끊어지면 3초 후 자동으로 재연결을 시도한다.

        구독 메시지 형식 (Upbit WebSocket API v1):
          [
            {"ticket": "unique-ticket-id"},
            {"type": "ticker", "codes": ["KRW-BTC", ...], "isOnlyRealtime": true},
            {"format": "DEFAULT"}
          ]
        """
        subscribe_data = [
            {"ticket": "coin-auto-trading"},
            {
                "type": "ticker",
                "codes": self._markets,
                "isOnlyRealtime": True,
            },
            {"format": "DEFAULT"},
        ]

        reconnect_delay = 3  # 재연결 대기 시간 (초)

        while self._ws_running:
            try:
                async with websockets.connect(
                    self.UPBIT_WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    # 구독 요청 전송
                    await ws.send(json.dumps(subscribe_data))
                    logger.info(
                        f"WebSocket 연결 성공: {len(self._markets)}개 마켓 구독"
                    )

                    # 메시지 수신 루프
                    while self._ws_running:
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=60
                            )
                            # Upbit WebSocket은 바이너리 또는 텍스트로 전송
                            if isinstance(raw, bytes):
                                data = json.loads(raw.decode("utf-8"))
                            else:
                                data = json.loads(raw)

                            self._handle_ws_message(data)

                        except asyncio.TimeoutError:
                            # 60초 동안 메시지가 없으면 연결 상태 점검
                            logger.warning(
                                "WebSocket 수신 타임아웃 (60초), 재연결 시도"
                            )
                            break

            except asyncio.CancelledError:
                # 태스크 취소 시 정상 종료
                logger.debug("WebSocket 핸들러 취소됨")
                break

            except ConnectionRefusedError:
                logger.error(
                    f"WebSocket 연결 거부, {reconnect_delay}초 후 재연결"
                )
                if self._ws_running:
                    await asyncio.sleep(reconnect_delay)

            except Exception as e:
                logger.error(
                    f"WebSocket 오류: {type(e).__name__}: {e}, "
                    f"{reconnect_delay}초 후 재연결"
                )
                if self._ws_running:
                    await asyncio.sleep(reconnect_delay)

    def _handle_ws_message(self, data: Dict[str, Any]) -> None:
        """
        WebSocket 메시지를 파싱하여 스냅샷을 갱신하고 콜백을 호출한다.

        Upbit ticker 메시지 주요 필드:
          - code: 마켓 코드 (예: "KRW-BTC")
          - trade_price: 현재 체결가
          - high_price: 당일 고가
          - low_price: 당일 저가
          - acc_trade_volume_24h: 24시간 누적 거래량
          - signed_change_rate: 전일 대비 등락률 (부호 포함)

        Args:
            data: 파싱된 WebSocket 메시지 딕셔너리
        """
        try:
            market = data.get("code", "")
            if not market:
                return

            snapshot = TickerSnapshot(
                market=market,
                current_price=float(data.get("trade_price", 0)),
                high_price=float(data.get("high_price", 0)),
                low_price=float(data.get("low_price", 0)),
                trade_volume=float(data.get("acc_trade_volume_24h", 0)),
                change_rate=float(data.get("signed_change_rate", 0)),
                timestamp=datetime.now(),
            )

            # 스냅샷 갱신 (스레드 안전)
            with self._snapshot_lock:
                self._snapshots[market] = snapshot

            # 등록된 콜백 호출
            for callback in self._tick_callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    logger.error(
                        f"틱 콜백 오류 ({callback.__name__ if hasattr(callback, '__name__') else 'unknown'}): "
                        f"{type(e).__name__}: {e}"
                    )

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"WebSocket 메시지 파싱 오류: {e}")
        except Exception as e:
            logger.error(f"WebSocket 메시지 처리 중 예기치 않은 오류: {e}")
