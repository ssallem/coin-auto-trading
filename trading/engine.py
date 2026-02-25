"""
트레이딩 엔진 (메인 엔진)

전체 자동매매 시스템의 핵심 오케스트레이터.
데이터 수집 -> 지표 계산 -> 전략 분석 -> 리스크 검증 -> 주문 실행의
전체 사이클을 관리한다.

데이터 흐름:
  1. DataCollector가 시세 데이터를 수집
  2. Indicators가 기술적 지표를 계산
  3. Strategy가 매매 신호를 생성
  4. RiskManager가 안전성을 검증
  5. OrderManager가 주문을 실행

중지 메커니즘:
  - threading.Event를 사용하여 외부에서 안전하게 중지 가능
  - Ctrl+C (KeyboardInterrupt) 시 graceful shutdown
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from api.upbit_client import Balance, UpbitClient
from config.settings import Settings
from data.collector import DataCollector
from data.indicators import Indicators
from strategies.base_strategy import BaseStrategy, Signal, SignalResult
from strategies.bollinger_strategy import BollingerStrategy
from strategies.ma_cross_strategy import MACrossStrategy
from strategies.rsi_strategy import RSIStrategy
from sync.supabase_sync import SupabaseSync
from trading.order_manager import OrderManager
from trading.risk_manager import RiskManager
from utils.logger import get_logger
from utils.notifier import get_notifier, init_notifier, send_notification

logger = get_logger(__name__)


class TradingEngine:
    """
    메인 트레이딩 엔진

    자동매매의 전체 생명주기를 관리하는 중앙 오케스트레이터.
    설정에 따라 지정된 마켓을 주기적으로 분석하고, 전략 신호에 따라
    자동으로 매매를 실행한다.

    사용법:
        settings = Settings.load()
        engine = TradingEngine(settings)
        engine.run()  # 무한 루프로 자동매매 시작
        # 다른 스레드에서: engine.stop()
    """

    def __init__(self, settings: Settings) -> None:
        """
        트레이딩 엔진을 초기화한다.

        모든 하위 컴포넌트를 설정 기반으로 초기화하고,
        전략은 run() 호출 시 lazy initialization 한다.

        Args:
            settings: 전역 설정 객체
        """
        self._settings = settings

        # 중지 신호 관리 (threading.Event)
        self._stop_event = threading.Event()

        # 사이클 카운터
        self._cycle_count = 0

        # ── Notifier 초기화 ──
        init_notifier(settings)

        # ── API 클라이언트 초기화 ──
        try:
            self._client = UpbitClient(
                access_key=settings.upbit_access_key,
                secret_key=settings.upbit_secret_key,
            )
        except ValueError as e:
            logger.critical(f"API 키 오류: {e}")
            raise RuntimeError(f"Upbit API 키를 확인하세요: {e}") from e

        # ── Supabase 동기화 초기화 ──
        self._sync = SupabaseSync()

        # ── 리스크 관리자 초기화 ──
        self._risk_manager = RiskManager(
            stop_loss_pct=settings.risk.stop_loss_pct,
            take_profit_pct=settings.risk.take_profit_pct,
            max_daily_loss=settings.risk.max_daily_loss,
            max_positions=settings.risk.max_positions,
            max_total_investment=settings.investment.max_total_investment,
            trailing_stop_enabled=settings.risk.trailing_stop_enabled,
            trailing_stop_pct=settings.risk.trailing_stop_pct,
        )

        # ── 주문 관리자 초기화 ──
        self._order_manager = OrderManager(
            client=self._client,
            risk_manager=self._risk_manager,
            per_trade_amount=settings.investment.per_trade_amount,
            min_order_amount=settings.investment.min_order_amount,
            sync=self._sync,
        )

        # ── 데이터 수집기 초기화 ──
        self._collector = DataCollector(
            client=self._client,
            markets=settings.trading.markets,
            timeframe=settings.trading.timeframe,
            candle_count=settings.trading.candle_count,
        )

        # ── 전략은 run() 호출 시 lazy initialization ──
        self._strategy: Optional[BaseStrategy] = None

        # ── 기존 보유 포지션 로드 (재시작 시 복구) ──
        self._load_existing_positions()

        logger.info(
            f"TradingEngine 초기화 완료: "
            f"전략=미설정 (run 시 자동 선택), "
            f"마켓={settings.trading.markets}, "
            f"폴링간격={settings.trading.poll_interval}초"
        )

    # ─────────────────────────────────────
    # 포지션 복구
    # ─────────────────────────────────────

    def _load_existing_positions(self) -> None:
        """
        Upbit API에서 현재 보유 중인 코인을 조회하여 RiskManager에 자동 등록한다.

        봇 재시작 시 메모리에서 사라진 포지션을 복구하여,
        실제 보유 자산과 봇의 인식 상태를 동기화한다.

        처리 로직:
          1. get_balances()로 전체 잔고 조회
          2. KRW가 아닌 코인(balance > 0)을 필터링
          3. 각 코인을 "KRW-{currency}" 마켓 코드로 변환
          4. avg_buy_price를 entry_price로 사용
          5. RiskManager.register_position() 호출

        에러 발생 시에도 봇 초기화는 계속 진행한다.
        """
        try:
            logger.info("기존 보유 포지션 로드 시작...")
            balances = self._client.get_balances()

            loaded_count = 0
            for balance in balances:
                # KRW는 스킵
                if balance.currency == "KRW":
                    continue

                # 보유 수량이 0보다 큰 코인만 처리
                if balance.balance <= 0:
                    continue

                # 마켓 코드 생성 (예: BTC → KRW-BTC)
                market = f"KRW-{balance.currency}"

                # avg_buy_price가 0인 경우 현재가로 폴백
                entry_price = balance.avg_buy_price
                if entry_price <= 0:
                    logger.warning(
                        f"{market}: avg_buy_price가 0 또는 음수 "
                        f"({entry_price}), 현재가로 대체 시도"
                    )
                    current_price = self._client.get_current_price(market)
                    if current_price and current_price > 0:
                        entry_price = current_price
                        logger.info(
                            f"{market}: 현재가 {entry_price:,.0f}를 "
                            f"entry_price로 사용"
                        )
                    else:
                        logger.error(
                            f"{market}: 현재가 조회 실패, 포지션 등록 스킵"
                        )
                        continue

                # 매수 총 금액 계산
                entry_amount = entry_price * balance.balance

                # RiskManager에 포지션 등록
                self._risk_manager.register_position(
                    market=market,
                    entry_price=entry_price,
                    volume=balance.balance,
                    entry_amount=entry_amount,
                    entry_time=datetime.now(),  # 정확한 매수 시각은 알 수 없으므로 현재 시각 사용
                )
                loaded_count += 1

            if loaded_count > 0:
                logger.info(
                    f"기존 포지션 로드 완료: {loaded_count}개 포지션 복구됨"
                )
            else:
                logger.info("기존 포지션 없음 (또는 KRW만 보유)")

        except Exception as e:
            logger.error(
                f"기존 포지션 로드 실패 (봇은 계속 실행됨): "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )

    # ─────────────────────────────────────
    # 전략 관리
    # ─────────────────────────────────────

    def _select_strategy(self, name: str) -> BaseStrategy:
        """
        전략 이름에 따라 전략 인스턴스를 생성한다.

        Args:
            name: 전략 이름 ("rsi", "ma_cross", "bollinger")
        Returns:
            BaseStrategy 구현 인스턴스
        Raises:
            ValueError: 알 수 없는 전략 이름
        """
        strategy_cfg = self._settings.strategy

        if name == "rsi":
            strategy = RSIStrategy(
                period=strategy_cfg.rsi.period,
                oversold=strategy_cfg.rsi.oversold,
                overbought=strategy_cfg.rsi.overbought,
            )
        elif name == "ma_cross":
            use_ema = strategy_cfg.ma_cross.ma_type.upper() == "EMA"
            strategy = MACrossStrategy(
                short_period=strategy_cfg.ma_cross.short_period,
                long_period=strategy_cfg.ma_cross.long_period,
                use_ema=use_ema,
            )
        elif name == "bollinger":
            strategy = BollingerStrategy(
                period=strategy_cfg.bollinger.period,
                std_dev=strategy_cfg.bollinger.std_dev,
            )
        else:
            raise ValueError(
                f"알 수 없는 전략: '{name}'. "
                f"유효한 전략: rsi, ma_cross, bollinger"
            )

        logger.info(f"전략 선택: {strategy.name}")
        return strategy

    def set_strategy(self, strategy: BaseStrategy) -> None:
        """
        매매 전략을 외부에서 교체한다.

        Args:
            strategy: BaseStrategy 구현 인스턴스
        """
        old_name = self._strategy.name if self._strategy else "None"
        self._strategy = strategy
        logger.info(f"전략 교체: {old_name} -> {strategy.name}")

    # ─────────────────────────────────────
    # 메인 실행
    # ─────────────────────────────────────

    def run(self) -> None:
        """
        자동매매를 시작한다 (무한 루프).

        매 사이클마다 다음을 수행한다:
          1. 모든 마켓의 시세 데이터 수집
          2. 기술적 지표 계산
          3. 보유 포지션의 리스크 체크 (손절/익절/트레일링)
          4. 전략 분석 및 신호에 따른 주문 실행

        _stop_event가 설정되거나 KeyboardInterrupt 발생 시 종료한다.
        """
        # 전략이 설정되지 않았으면 config에서 자동 선택 (lazy initialization)
        if self._strategy is None:
            self._strategy = self._select_strategy(
                self._settings.strategy.active
            )

        # 중지 이벤트 초기화
        self._stop_event.clear()
        poll_interval = self._settings.trading.poll_interval

        logger.info("=" * 60)
        logger.info("자동매매 엔진 시작")
        logger.info(f"  전략: {self._strategy.name}")
        logger.info(f"  마켓: {self._settings.trading.markets}")
        logger.info(f"  시간프레임: {self._settings.trading.timeframe}")
        logger.info(f"  폴링 간격: {poll_interval}초")
        logger.info(f"  손절: {self._settings.risk.stop_loss_pct}%")
        logger.info(f"  익절: {self._settings.risk.take_profit_pct}%")
        logger.info(
            f"  트레일링 스탑: "
            f"{'ON' if self._settings.risk.trailing_stop_enabled else 'OFF'} "
            f"({self._settings.risk.trailing_stop_pct}%)"
        )
        logger.info("=" * 60)

        send_notification(
            f"[자동매매 시작]\n"
            f"전략: {self._strategy.name}\n"
            f"마켓: {', '.join(self._settings.trading.markets)}\n"
            f"시간프레임: {self._settings.trading.timeframe}"
        )

        try:
            while not self._stop_event.is_set():
                self._cycle_count += 1
                self._execute_cycle()

                # poll_interval 동안 대기하되, stop_event가 설정되면 즉시 깨어남
                self._stop_event.wait(timeout=poll_interval)

        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨 (Ctrl+C)")
        except Exception as e:
            logger.critical(f"엔진 치명적 오류: {e}", exc_info=True)
            # 치명적 오류 알림
            notifier = get_notifier()
            if notifier:
                notifier.send_error_alert(
                    f"엔진 치명적 오류: {type(e).__name__}: {e}"
                )
        finally:
            self.stop()

    def stop(self) -> None:
        """
        자동매매를 안전하게 중지한다.

        _stop_event를 설정하여 메인 루프를 종료시키고,
        WebSocket 연결을 닫고, 거래 요약을 출력한다.
        """
        self._stop_event.set()
        self._collector.stop_websocket()

        logger.info("자동매매 엔진 종료")
        self._print_summary()

        send_notification(
            f"[자동매매 종료]\n"
            f"총 사이클: {self._cycle_count}\n"
            f"총 실현 손익: {self._order_manager.get_total_pnl():,.0f} KRW"
        )

    # ─────────────────────────────────────
    # 매매 사이클
    # ─────────────────────────────────────

    def _execute_cycle(self) -> None:
        """
        한 번의 매매 사이클을 실행한다.

        처리 순서:
          1. 모든 마켓의 캔들 데이터 수집
          2. 각 마켓에 대해:
             a. 기술적 지표 추가
             b. 보유 포지션 리스크 체크 (손절/익절/트레일링)
             c. 전략 분석 → 매매 신호 생성
             d. 신호에 따른 주문 실행

        개별 마켓 처리 중 에러가 발생해도 다른 마켓은 계속 처리된다.
        """
        try:
            logger.debug(f"--- 사이클 #{self._cycle_count} 시작 ---")

            for market in self._settings.trading.markets:
                try:
                    self._process_market(market)
                except Exception as e:
                    logger.error(
                        f"마켓 {market} 처리 중 오류: "
                        f"{type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    # 마켓별 에러는 알림 전송 후 계속 진행
                    notifier = get_notifier()
                    if notifier:
                        notifier.send_error_alert(
                            f"마켓 {market} 처리 오류: "
                            f"{type(e).__name__}: {e}"
                        )

            # ── Supabase 동기화 ──
            self._sync_account_if_needed()
            self._process_pending_orders()

            logger.debug(f"--- 사이클 #{self._cycle_count} 완료 ---")

        except Exception as e:
            logger.error(
                f"사이클 #{self._cycle_count} 전체 오류: {e}",
                exc_info=True,
            )
            notifier = get_notifier()
            if notifier:
                notifier.send_error_alert(
                    f"사이클 #{self._cycle_count} 오류: "
                    f"{type(e).__name__}: {e}"
                )

    def _process_market(self, market: str) -> None:
        """
        개별 마켓에 대해 데이터 수집 → 지표 계산 → 리스크 체크 → 전략 분석 → 주문을 수행한다.

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
        """
        # 1. 캔들 데이터 수집
        df = self._collector.get_candles(market)
        if df is None or df.empty:
            logger.warning(f"[{market}] 캔들 데이터 수집 실패, 스킵")
            return

        # 2. 기술적 지표 추가
        df = Indicators.add_all_indicators(df)
        if df.empty:
            logger.warning(f"[{market}] 지표 계산 실패, 스킵")
            return

        # 3. 현재가 조회
        current_price = self._client.get_current_price(market)
        if current_price is None:
            logger.warning(f"[{market}] 현재가 조회 실패, 스킵")
            return

        # 4. 보유 포지션 리스크 체크 (손절/익절/트레일링 스탑)
        position = self._risk_manager.get_position(market)
        if position is not None:
            action = self._risk_manager.check_position(market, current_price)
            if action is not None:
                # 리스크 이벤트 발동 → 자동 매도
                reason_map = {
                    "stop_loss": "손절 발동",
                    "take_profit": "익절 발동",
                    "trailing_stop": "트레일링 스탑 발동",
                }
                reason = reason_map.get(action, action)
                pnl_pct = position.unrealized_pnl_pct(current_price)

                logger.info(
                    f"[자동 매도] {market}: {reason} "
                    f"(손익률: {pnl_pct:+.2f}%)"
                )

                # 매도 실행
                record = self._order_manager.execute_sell(
                    market, reason=reason
                )

                # 리스크 알림 전송
                notifier = get_notifier()
                if notifier:
                    notifier.send_risk_alert(
                        market=market,
                        alert_type=action,
                        pnl_pct=pnl_pct,
                    )

                # 전략 콜백
                if record and self._strategy:
                    self._strategy.on_position_closed(market, pnl_pct)

                # 리스크 이벤트로 매도한 경우, 이 마켓은 이번 사이클에서 추가 분석하지 않음
                return

        # 5. 전략 분석 → 매매 신호 생성
        signal_result = self._strategy.analyze(market, df, current_price)
        logger.debug(
            f"[{market}] 신호: {signal_result.signal.value}, "
            f"확신도: {signal_result.confidence:.2f}, "
            f"사유: {signal_result.reason}"
        )

        # 6. 신호에 따른 매매 실행
        if signal_result.signal == Signal.BUY:
            record = self._order_manager.execute_buy(
                market, reason=signal_result.reason
            )
            if record and self._strategy:
                self._strategy.on_position_opened(
                    market, record.price, record.amount
                )

        elif signal_result.signal == Signal.SELL:
            # 보유 중인 경우에만 매도
            sell_position = self._risk_manager.get_position(market)
            if sell_position is not None:
                record = self._order_manager.execute_sell(
                    market, reason=signal_result.reason
                )
                if record and self._strategy:
                    self._strategy.on_position_closed(market, record.pnl_pct)

    # ─────────────────────────────────────
    # Supabase 동기화
    # ─────────────────────────────────────

    def _sync_account_if_needed(self) -> None:
        """
        계좌 잔고를 조회하여 Supabase에 스냅샷을 전송한다.

        push_account_snapshot 내부에서 30초 주기를 관리하므로
        매 사이클 호출해도 과도한 전송은 발생하지 않는다.
        동기화 실패 시에도 매매 로직에 영향을 주지 않는다.
        """
        if not self._sync.enabled:
            return

        try:
            balances = self._client.get_balances()

            krw_balance = 0.0
            krw_locked = 0.0
            holdings = []

            for b in balances:
                if b.currency == "KRW":
                    krw_balance = b.balance
                    krw_locked = b.locked
                else:
                    holdings.append({
                        "currency": b.currency,
                        "balance": str(b.balance),
                        "locked": str(b.locked),
                        "avg_buy_price": str(b.avg_buy_price),
                        "unit_currency": b.unit_currency,
                    })

            self._sync.push_account_snapshot(
                krw_balance=krw_balance,
                krw_locked=krw_locked,
                holdings=holdings,
            )
        except Exception as e:
            logger.debug(f"계좌 동기화 실패 (무시): {e}")

    def _process_pending_orders(self) -> None:
        """
        웹 대시보드에서 생성된 대기 주문을 조회하고 실행한다.

        각 주문의 side/ord_type 조합에 따라 적절한 API를 호출한다:
          - bid + price  : 시장가 매수 (금액 지정)
          - ask + market : 시장가 매도 (수량 지정)
          - bid + limit  : 지정가 매수
          - ask + limit  : 지정가 매도

        실행 결과에 따라 pending_orders 상태를 갱신하고
        order_history에 이력을 기록한다.
        """
        if not self._sync.enabled:
            return

        try:
            pending = self._sync.fetch_pending_orders()
        except Exception as e:
            logger.debug(f"대기 주문 조회 실패 (무시): {e}")
            return

        for order in pending:
            order_id = order.get("id")
            market = order.get("market", "")
            side = order.get("side", "")
            ord_type = order.get("ord_type", "")
            price = order.get("price")
            volume = order.get("volume")

            try:
                self._sync.mark_pending_processing(order_id)

                order_result = None

                if side == "bid" and ord_type == "price":
                    # 시장가 매수 (금액 지정)
                    order_result = self._client.buy_market_order(
                        market, float(price)
                    )
                elif side == "ask" and ord_type == "market":
                    # 시장가 매도 (수량 지정)
                    order_result = self._client.sell_market_order(
                        market, float(volume)
                    )
                elif side == "bid" and ord_type == "limit":
                    # 지정가 매수
                    order_result = self._client.buy_limit_order(
                        market, float(price), float(volume)
                    )
                elif side == "ask" and ord_type == "limit":
                    # 지정가 매도
                    order_result = self._client.sell_limit_order(
                        market, float(price), float(volume)
                    )
                else:
                    raise ValueError(
                        f"지원하지 않는 주문 유형: side={side}, ord_type={ord_type}"
                    )

                if order_result is None:
                    raise RuntimeError("주문 API가 None을 반환했습니다.")

                # 성공
                self._sync.mark_pending_done(order_id, order_result.uuid)

                # bid+price: 주문 금액이 곧 amount
                if side == "bid":
                    amount_val = float(price) if price else 0.0
                # ask: 체결 후 현재가 * 수량으로 추정 (정확한 체결가는 알 수 없음)
                else:
                    current_price = self._client.get_current_price(market)
                    amount_val = float(volume) * current_price if current_price and volume else 0.0

                self._sync.push_order_history(
                    upbit_uuid=order_result.uuid,
                    market=market,
                    side=side,
                    ord_type=ord_type,
                    price=float(price) if price else 0.0,
                    volume=float(volume) if volume else 0.0,
                    amount=amount_val,
                    source="web",
                    pending_order_id=order_id,
                )

                logger.info(
                    f"대기 주문 #{order_id} 실행 완료: "
                    f"{market} {side} {ord_type}"
                )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                self._sync.mark_pending_failed(order_id, error_msg)
                logger.warning(
                    f"대기 주문 #{order_id} 실행 실패: {error_msg}"
                )

    # ─────────────────────────────────────
    # 요약 출력
    # ─────────────────────────────────────

    def _print_summary(self) -> None:
        """종료 시 거래 요약을 출력한다."""
        history = self._order_manager.get_trade_history()
        total_pnl = self._order_manager.get_total_pnl()
        stats = self._risk_manager.get_daily_stats()

        # 매수/매도 건수 분리
        buy_count = sum(1 for t in history if t.side == "buy")
        sell_count = sum(1 for t in history if t.side == "sell")

        # 남아있는 포지션
        remaining_positions = self._risk_manager.get_all_positions()

        logger.info("=" * 60)
        logger.info("거래 요약")
        logger.info(f"  총 사이클: {self._cycle_count}")
        logger.info(f"  총 거래 수: {len(history)} (매수 {buy_count}건, 매도 {sell_count}건)")
        logger.info(f"  총 실현 손익: {total_pnl:,.0f} KRW")
        logger.info(f"  당일 실현 손익: {stats.realized_pnl:,.0f} KRW")
        logger.info(f"  당일 수익 거래: {stats.winning_trades}건")
        logger.info(f"  당일 손실 거래: {stats.losing_trades}건")
        if remaining_positions:
            logger.info(f"  미청산 포지션: {list(remaining_positions.keys())}")
        else:
            logger.info("  미청산 포지션: 없음")
        logger.info("=" * 60)
