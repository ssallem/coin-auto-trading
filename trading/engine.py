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
from typing import Dict, Optional

import pandas as pd

from api.upbit_client import UpbitClient
from config.settings import Settings
from data.collector import DataCollector
from data.indicators import Indicators
from strategies.base_strategy import BaseStrategy, Signal, SignalResult
from strategies.bollinger_strategy import BollingerStrategy
from strategies.ma_cross_strategy import MACrossStrategy
from strategies.rsi_strategy import RSIStrategy
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

        logger.info(
            f"TradingEngine 초기화 완료: "
            f"전략=미설정 (run 시 자동 선택), "
            f"마켓={settings.trading.markets}, "
            f"폴링간격={settings.trading.poll_interval}초"
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
