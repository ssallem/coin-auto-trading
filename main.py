"""
CoinAutoTrading Bot - 메인 엔트리포인트 (CLI)

서브커맨드:
  trade     실시간 자동매매를 시작한다.
  backtest  과거 데이터를 이용한 백테스트를 실행한다.
  check     설정 파일 및 API 연결 상태를 확인한다.

사용 예:
  python main.py trade
  python main.py backtest --market KRW-BTC --interval day --count 200
  python main.py check
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import traceback

# ── 프로젝트 루트를 sys.path 에 추가하여 패키지 import 해결 ──
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import Settings
from utils.logger import setup_logger, get_logger


# ─────────────────────────────────────────────
# 전략 팩토리
# ─────────────────────────────────────────────

def create_strategy(settings: Settings, strategy_name: str | None = None):
    """
    설정(또는 CLI에서 지정한 이름)에 따라 전략 인스턴스를 생성한다.

    Args:
        settings: Settings 싱글턴 인스턴스
        strategy_name: 전략 이름 (None이면 config에서 읽음)
    Returns:
        BaseStrategy 구현 인스턴스
    Raises:
        ValueError: 알 수 없는 전략 이름
    """
    active = strategy_name or settings.strategy.active

    if active == "rsi":
        from strategies.rsi_strategy import RSIStrategy
        return RSIStrategy(
            period=settings.strategy.rsi.period,
            oversold=settings.strategy.rsi.oversold,
            overbought=settings.strategy.rsi.overbought,
        )
    elif active == "ma_cross":
        from strategies.ma_cross_strategy import MACrossStrategy
        return MACrossStrategy(
            short_period=settings.strategy.ma_cross.short_period,
            long_period=settings.strategy.ma_cross.long_period,
            use_ema=(settings.strategy.ma_cross.ma_type.upper() == "EMA"),
        )
    elif active == "bollinger":
        from strategies.bollinger_strategy import BollingerStrategy
        return BollingerStrategy(
            period=settings.strategy.bollinger.period,
            std_dev=settings.strategy.bollinger.std_dev,
        )
    else:
        raise ValueError(f"알 수 없는 전략: {active}")


# ─────────────────────────────────────────────
# 서브커맨드: trade
# ─────────────────────────────────────────────

def cmd_trade(args: argparse.Namespace) -> None:
    """실시간 자동매매를 시작한다."""
    try:
        settings = Settings.load()
    except Exception as e:
        print(f"[오류] 설정 파일 로드 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # 로거를 먼저 설정
    setup_logger(
        level=settings.logging.level,
        log_file=settings.logging.file,
        max_size_mb=settings.logging.max_size_mb,
        backup_count=settings.logging.backup_count,
    )
    logger = get_logger(__name__)

    # 알림 모듈 초기화
    from utils.notifier import init_notifier
    init_notifier(settings)

    # 설정 검증
    errors = settings.validate()
    if errors:
        print("설정 오류가 있습니다:")
        for err in errors:
            print(f"  - {err}")
        print("\n.env 파일과 config/config.yaml을 확인해주세요.")
        sys.exit(1)

    # 전략 생성
    try:
        strategy = create_strategy(settings)
    except ValueError as e:
        logger.error("전략 생성 실패: %s", e)
        sys.exit(1)

    # 트레이딩 엔진 생성 및 실행
    from trading.engine import TradingEngine

    engine = TradingEngine(settings)
    engine.set_strategy(strategy)

    # Ctrl+C graceful shutdown 핸들러
    def _shutdown_handler(signum, frame):
        logger.info("종료 신호 수신 (Ctrl+C). 자동매매를 안전하게 종료합니다...")
        engine.stop()

    signal.signal(signal.SIGINT, _shutdown_handler)

    logger.info("자동매매를 시작합니다. 종료하려면 Ctrl+C를 누르세요.")

    try:
        engine.run()
    except KeyboardInterrupt:
        # run() 내부에서도 처리하지만, 이중 안전장치
        logger.info("사용자에 의해 중단됨 (Ctrl+C)")
    except Exception as e:
        logger.critical("자동매매 엔진 치명적 오류: %s", e, exc_info=True)
        sys.exit(1)


# ─────────────────────────────────────────────
# 서브커맨드: backtest
# ─────────────────────────────────────────────

def cmd_backtest(args: argparse.Namespace) -> None:
    """백테스트를 실행한다."""
    try:
        settings = Settings.load()
    except Exception as e:
        print(f"[오류] 설정 파일 로드 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # 로거를 먼저 설정
    setup_logger(
        level=settings.logging.level,
        log_file=settings.logging.file,
        max_size_mb=settings.logging.max_size_mb,
        backup_count=settings.logging.backup_count,
    )
    logger = get_logger(__name__)

    # CLI 인자 또는 config 기본값
    market = args.market or settings.trading.markets[0]
    interval = args.interval or settings.trading.timeframe
    count = args.count or settings.trading.candle_count
    capital = args.capital or settings.backtest.initial_capital
    strategy_name = args.strategy  # None이면 config에서

    # 전략 생성
    try:
        strategy = create_strategy(settings, strategy_name=strategy_name)
    except ValueError as e:
        logger.error("전략 생성 실패: %s", e)
        sys.exit(1)

    # UpbitClient 생성 (API 키가 없어도 get_ohlcv는 공개 API이므로 동작 가능)
    from api.upbit_client import UpbitClient
    from backtesting.engine import BacktestEngine

    try:
        client = UpbitClient(
            access_key=settings.upbit_access_key or "backtest_dummy",
            secret_key=settings.upbit_secret_key or "backtest_dummy",
        )
    except ValueError:
        # API 키가 비어있으면 더미로 생성 시도 - pyupbit 내부에서 처리
        print(
            "[경고] API 키가 설정되지 않았습니다. "
            "공개 API로 데이터를 조회합니다.",
            file=sys.stderr,
        )
        # pyupbit.get_ohlcv는 인증 불필요하므로 더미 클라이언트로 진행
        client = UpbitClient(
            access_key="dummy_access_key_for_backtest",
            secret_key="dummy_secret_key_for_backtest",
        )

    engine = BacktestEngine(
        client=client,
        strategy=strategy,
        initial_capital=capital,
        commission_rate=settings.backtest.commission_rate,
    )

    print(f"\n백테스트 실행 중...")
    print(f"  마켓: {market}")
    print(f"  전략: {strategy.name}")
    print(f"  간격: {interval}")
    print(f"  캔들 수: {count}")
    print(f"  초기 자본: {capital:,.0f} KRW")
    print(f"  수수료율: {settings.backtest.commission_rate * 100:.3f}%")

    try:
        result = engine.run(
            market=market,
            interval=interval,
            count=count,
        )
        print(result.summary())
    except ValueError as e:
        logger.error("백테스트 실행 실패: %s", e)
        print(f"\n[오류] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error("백테스트 중 예기치 않은 오류: %s", e, exc_info=True)
        print(f"\n[오류] 백테스트 실패: {e}", file=sys.stderr)
        sys.exit(1)


# ─────────────────────────────────────────────
# 서브커맨드: check
# ─────────────────────────────────────────────

def cmd_check(args: argparse.Namespace) -> None:
    """설정 및 API 연결 상태를 확인한다."""
    print("=" * 55)
    print("  설정 및 API 연결 확인")
    print("=" * 55)

    # ── 1단계: 설정 파일 로드 ──
    print("\n[1/3] 설정 파일 로드...")
    try:
        settings = Settings.load()
        print("  -> 성공: 설정 파일을 정상적으로 로드했습니다.")
    except Exception as e:
        print(f"  -> 실패: 설정 파일 로드 오류 - {e}")
        sys.exit(1)

    # ── 2단계: 설정 검증 ──
    print("\n[2/3] 설정 유효성 검증...")
    errors = settings.validate()
    if errors:
        print("  -> 검증 실패:")
        for err in errors:
            print(f"     - {err}")
    else:
        print("  -> 성공: 모든 설정이 유효합니다.")

    # 주요 설정값 출력
    print(f"\n  [설정 요약]")
    print(f"    마켓:         {settings.trading.markets}")
    print(f"    전략:         {settings.strategy.active}")
    print(f"    타임프레임:   {settings.trading.timeframe}")
    print(f"    폴링 간격:    {settings.trading.poll_interval}초")
    print(f"    1회 투자금:   {settings.investment.per_trade_amount:,} KRW")
    print(f"    손절:         {settings.risk.stop_loss_pct}%")
    print(f"    익절:         {settings.risk.take_profit_pct}%")

    # ── 3단계: API 키 테스트 (잔고 조회) ──
    print("\n[3/3] Upbit API 연결 테스트...")

    if not settings.upbit_access_key or not settings.upbit_secret_key:
        print("  -> 건너뜀: API 키가 설정되지 않았습니다.")
        print("     .env 파일에 UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY를 설정하세요.")
        if errors:
            sys.exit(1)
        return

    placeholder_values = {"your_access_key_here", "your_secret_key_here"}
    if (
        settings.upbit_access_key in placeholder_values
        or settings.upbit_secret_key in placeholder_values
    ):
        print("  -> 건너뜀: API 키가 플레이스홀더 값입니다.")
        print("     .env 파일의 API 키를 실제 값으로 교체하세요.")
        if errors:
            sys.exit(1)
        return

    from api.upbit_client import UpbitClient

    try:
        client = UpbitClient(
            access_key=settings.upbit_access_key,
            secret_key=settings.upbit_secret_key,
        )
        balances = client.get_balances()
        print("  -> 성공: API 연결 정상")

        # KRW 잔고 표시
        krw_balance = 0.0
        for b in balances:
            if b.currency == "KRW":
                krw_balance = b.balance
                break
        print(f"     KRW 잔고: {krw_balance:,.0f} 원")

        # 보유 코인 목록
        coins = [b for b in balances if b.currency != "KRW" and b.balance > 0]
        if coins:
            print(f"     보유 코인: {len(coins)}종")
            for c in coins:
                print(
                    f"       - {c.currency}: {c.balance:.8f} "
                    f"(평균 매수가: {c.avg_buy_price:,.0f})"
                )
        else:
            print("     보유 코인: 없음")

    except ConnectionError as e:
        print(f"  -> 실패: API 연결 오류 - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"  -> 실패: API 클라이언트 초기화 오류 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  -> 실패: 예기치 않은 오류 - {e}")
        traceback.print_exc()
        sys.exit(1)

    # 최종 결과
    print(f"\n{'=' * 55}")
    if errors:
        print("  결과: 일부 설정에 문제가 있습니다. 위의 오류를 확인하세요.")
        sys.exit(1)
    else:
        print("  결과: 모든 점검 통과! 자동매매를 시작할 수 있습니다.")
    print(f"{'=' * 55}")


# ─────────────────────────────────────────────
# CLI 파서 구성
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """argparse 기반 CLI 파서를 구성한다."""
    parser = argparse.ArgumentParser(
        prog="CoinAutoTrading",
        description="Upbit 자동매매 봇 - 자동매매, 백테스트, 설정 확인",
    )

    subparsers = parser.add_subparsers(
        title="명령어",
        description="실행할 명령어를 선택하세요",
        dest="command",
    )

    # ── trade 서브커맨드 ──
    trade_parser = subparsers.add_parser(
        "trade",
        help="실시간 자동매매를 시작합니다",
        description="설정 파일 기반으로 실시간 자동매매를 시작합니다. Ctrl+C로 안전하게 종료할 수 있습니다.",
    )
    trade_parser.set_defaults(func=cmd_trade)

    # ── backtest 서브커맨드 ──
    bt_parser = subparsers.add_parser(
        "backtest",
        help="과거 데이터 기반 백테스트를 실행합니다",
        description="과거 캔들 데이터를 이용하여 매매 전략의 성과를 시뮬레이션합니다.",
    )
    bt_parser.add_argument(
        "--market",
        type=str,
        default=None,
        help="백테스트 대상 마켓 코드 (기본: config에서 읽음, 예: KRW-BTC)",
    )
    bt_parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="캔들 간격 (기본: config에서 읽음, 예: day, minute60)",
    )
    bt_parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="캔들 개수 (기본: config에서 읽음, 예: 200)",
    )
    bt_parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["rsi", "ma_cross", "bollinger"],
        help="사용할 전략 (기본: config에서 읽음)",
    )
    bt_parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="초기 자본금 KRW (기본: config에서 읽음, 예: 1000000)",
    )
    bt_parser.set_defaults(func=cmd_backtest)

    # ── check 서브커맨드 ──
    check_parser = subparsers.add_parser(
        "check",
        help="설정 파일 및 API 연결 상태를 확인합니다",
        description="설정 파일의 유효성을 검증하고, Upbit API 키의 정상 동작 여부를 테스트합니다.",
    )
    check_parser.set_defaults(func=cmd_check)

    return parser


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main() -> None:
    """메인 함수 - CLI 파싱 후 적절한 서브커맨드를 실행한다."""
    parser = build_parser()
    args = parser.parse_args()

    # 서브커맨드가 지정되지 않은 경우 도움말 출력
    if not hasattr(args, "func") or args.func is None:
        parser.print_help()
        sys.exit(0)

    # 서브커맨드 실행
    args.func(args)


if __name__ == "__main__":
    main()
