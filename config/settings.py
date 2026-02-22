"""
설정 관리 모듈

config.yaml과 .env 파일을 로드하여 애플리케이션 전역 설정을 관리한다.
싱글턴 패턴을 적용하여 설정 객체가 하나만 존재하도록 보장한다.

사용법:
    from config.settings import Settings

    settings = Settings.load()
    print(settings.trading.markets)
    print(settings.upbit_access_key)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


# 프로젝트 루트 경로 (config/ 의 상위 디렉토리)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────
# 설정 데이터 클래스들
# ─────────────────────────────────────────────

@dataclass
class TradingConfig:
    """거래 대상 설정"""
    markets: List[str] = field(default_factory=lambda: ["KRW-BTC"])
    poll_interval: int = 60
    timeframe: str = "minute60"
    candle_count: int = 200


@dataclass
class InvestmentConfig:
    """투자 금액 설정"""
    max_total_investment: int = 1_000_000
    per_trade_amount: int = 100_000
    min_order_amount: int = 5_000


@dataclass
class RiskConfig:
    """리스크 관리 설정"""
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 5.0
    max_daily_loss: int = 50_000
    max_positions: int = 3
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 2.0


@dataclass
class RSIStrategyConfig:
    """RSI 전략 파라미터"""
    period: int = 14
    oversold: int = 30
    overbought: int = 70


@dataclass
class MACrossStrategyConfig:
    """이동평균 교차 전략 파라미터"""
    short_period: int = 5
    long_period: int = 20
    ma_type: str = "EMA"


@dataclass
class BollingerStrategyConfig:
    """볼린저밴드 전략 파라미터"""
    period: int = 20
    std_dev: float = 2.0


@dataclass
class StrategyConfig:
    """전략 설정"""
    active: str = "rsi"
    rsi: RSIStrategyConfig = field(default_factory=RSIStrategyConfig)
    ma_cross: MACrossStrategyConfig = field(default_factory=MACrossStrategyConfig)
    bollinger: BollingerStrategyConfig = field(default_factory=BollingerStrategyConfig)


@dataclass
class BacktestConfig:
    """백테스팅 설정"""
    period_days: int = 30
    start_date: str = ""
    end_date: str = ""
    initial_capital: int = 1_000_000
    commission_rate: float = 0.0005


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    file: str = "logs/trading.log"
    max_size_mb: int = 10
    backup_count: int = 5


@dataclass
class NotificationConfig:
    """알림 설정"""
    enabled: bool = False
    channel: str = "telegram"
    events: List[str] = field(default_factory=lambda: [
        "order_executed",
        "stop_loss_triggered",
        "take_profit_triggered",
        "daily_report",
        "error",
    ])


# ─────────────────────────────────────────────
# 메인 Settings 클래스 (싱글턴)
# ─────────────────────────────────────────────

class Settings:
    """
    애플리케이션 전역 설정 관리 (싱글턴)

    .env 파일에서 API 키를 로드하고, config.yaml에서 거래/리스크/전략 설정을
    로드하여 타입 안전한 dataclass 객체로 매핑한다.

    사용법:
        settings = Settings.load()
        print(settings.trading.markets)
        print(settings.upbit_access_key)
    """

    _instance: Optional[Settings] = None

    def __init__(self) -> None:
        # ── .env 파일 로드 ──
        env_path = PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=env_path)

        # ── API 키 (환경변수에서 로드) ──
        self.upbit_access_key: str = os.getenv("UPBIT_ACCESS_KEY", "")
        self.upbit_secret_key: str = os.getenv("UPBIT_SECRET_KEY", "")
        self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
        self.slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")

        # ── 설정 로드: Supabase 우선, yaml fallback ──
        raw: Optional[Dict[str, Any]] = None
        raw = self._load_from_supabase()
        if raw is None:
            print("[정보] Supabase 미설정 또는 로드 실패, config.yaml로 폴백합니다.")
            config_path = PROJECT_ROOT / "config" / "config.yaml"
            raw = self._load_yaml(config_path)
        else:
            print("[정보] Supabase에서 설정을 로드했습니다.")

        # ── 각 섹션을 dataclass 인스턴스로 매핑 ──
        self.trading = self._load_dataclass(
            TradingConfig, raw.get("trading", {})
        )
        self.investment = self._load_dataclass(
            InvestmentConfig, raw.get("investment", {})
        )

        # risk 섹션 특별 처리 (Supabase 중첩 구조 → dataclass 플랫 구조)
        risk_raw = dict(raw.get("risk", {}))
        trailing_raw = risk_raw.pop("trailing_stop", {})
        if trailing_raw and isinstance(trailing_raw, dict):
            risk_raw["trailing_stop_enabled"] = trailing_raw.get("enabled", False)
            risk_raw["trailing_stop_pct"] = trailing_raw.get("pct", 2.0)
        self.risk = self._load_dataclass(
            RiskConfig, risk_raw
        )
        self.backtest = self._load_dataclass(
            BacktestConfig, raw.get("backtest", {})
        )
        self.logging = self._load_dataclass(
            LoggingConfig, raw.get("logging", {})
        )
        self.notification = self._load_dataclass(
            NotificationConfig, raw.get("notification", {})
        )

        # ── 전략 설정은 중첩 구조이므로 별도 처리 ──
        strat_raw = raw.get("strategy", {})
        self.strategy = StrategyConfig(
            active=strat_raw.get("active", "rsi"),
            rsi=self._load_dataclass(
                RSIStrategyConfig, strat_raw.get("rsi", {})
            ),
            ma_cross=self._load_dataclass(
                MACrossStrategyConfig, strat_raw.get("ma_cross", {})
            ),
            bollinger=self._load_dataclass(
                BollingerStrategyConfig, strat_raw.get("bollinger", {})
            ),
        )

    # ─────────────────────────────────────────
    # 클래스 메서드: 싱글턴 로드
    # ─────────────────────────────────────────

    @classmethod
    def load(cls, force_reload: bool = False) -> Settings:
        """
        설정 싱글턴 인스턴스를 반환한다.

        Args:
            force_reload: True이면 기존 인스턴스를 버리고 새로 로드한다.
        Returns:
            Settings 싱글턴 인스턴스
        """
        if cls._instance is None or force_reload:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """싱글턴 인스턴스를 초기화한다 (주로 테스트용)."""
        cls._instance = None

    # ─────────────────────────────────────────
    # Supabase 로드
    # ─────────────────────────────────────────

    @staticmethod
    def _load_from_supabase() -> Optional[Dict[str, Any]]:
        """Supabase에서 설정을 로드한다. 실패 시 None."""
        try:
            from config.supabase_loader import load_config_from_supabase
            return load_config_from_supabase()
        except ImportError:
            return None

    # ─────────────────────────────────────────
    # YAML 로드
    # ─────────────────────────────────────────

    @staticmethod
    def _load_yaml(config_path: Path) -> Dict[str, Any]:
        """
        YAML 설정 파일을 로드한다.

        Args:
            config_path: config.yaml 파일 경로
        Returns:
            파싱된 딕셔너리 (파일 없으면 빈 딕셔너리)
        Raises:
            yaml.YAMLError: YAML 파싱 실패 시
        """
        if not config_path.exists():
            print(
                f"[경고] 설정 파일을 찾을 수 없습니다: {config_path}\n"
                f"기본값을 사용합니다.",
                file=sys.stderr,
            )
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"config.yaml 파싱 오류: {e}"
            ) from e

    # ─────────────────────────────────────────
    # 데이터클래스 변환
    # ─────────────────────────────────────────

    @staticmethod
    def _load_dataclass(dc_class: type, data: Dict[str, Any]):
        """
        딕셔너리에서 데이터클래스 인스턴스를 생성한다.

        데이터클래스에 정의된 필드만 추출하고, 나머지 키는 무시한다.
        이를 통해 YAML에 불필요한 키가 있어도 에러 없이 로드된다.

        Args:
            dc_class: 대상 데이터클래스 타입
            data: 원본 딕셔너리 데이터
        Returns:
            데이터클래스 인스턴스
        """
        if not data:
            return dc_class()

        # 데이터클래스 필드에 해당하는 키만 추출
        valid_keys = {f.name for f in dc_class.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return dc_class(**filtered)

    # ─────────────────────────────────────────
    # 설정 검증
    # ─────────────────────────────────────────

    def validate(self) -> List[str]:
        """
        설정의 유효성을 검사하고 경고/오류 메시지 목록을 반환한다.

        Returns:
            에러 메시지 목록 (빈 목록이면 유효한 설정)
        """
        errors: List[str] = []

        # ── API 키 검증 ──
        placeholder_values = {"your_access_key_here", "your_secret_key_here", ""}
        if self.upbit_access_key in placeholder_values:
            errors.append(
                "[필수] UPBIT_ACCESS_KEY가 설정되지 않았습니다. "
                ".env 파일을 확인하세요."
            )
        if self.upbit_secret_key in placeholder_values:
            errors.append(
                "[필수] UPBIT_SECRET_KEY가 설정되지 않았습니다. "
                ".env 파일을 확인하세요."
            )

        # ── 투자 금액 검증 ──
        if self.investment.per_trade_amount < self.investment.min_order_amount:
            errors.append(
                f"[설정 오류] per_trade_amount({self.investment.per_trade_amount:,})"
                f" < min_order_amount({self.investment.min_order_amount:,})"
            )
        if self.investment.per_trade_amount > self.investment.max_total_investment:
            errors.append(
                f"[설정 오류] per_trade_amount({self.investment.per_trade_amount:,})"
                f" > max_total_investment({self.investment.max_total_investment:,})"
            )
        if self.investment.max_total_investment <= 0:
            errors.append("[설정 오류] max_total_investment는 0보다 커야 합니다.")

        # ── 리스크 설정 검증 ──
        if self.risk.stop_loss_pct <= 0:
            errors.append("[설정 오류] stop_loss_pct는 0보다 커야 합니다.")
        if self.risk.take_profit_pct <= 0:
            errors.append("[설정 오류] take_profit_pct는 0보다 커야 합니다.")
        if self.risk.max_positions < 1:
            errors.append("[설정 오류] max_positions는 1 이상이어야 합니다.")
        if self.risk.max_daily_loss <= 0:
            errors.append("[설정 오류] max_daily_loss는 0보다 커야 합니다.")
        if self.risk.trailing_stop_enabled and self.risk.trailing_stop_pct <= 0:
            errors.append(
                "[설정 오류] trailing_stop이 활성화되어 있지만 "
                "trailing_stop_pct가 0 이하입니다."
            )

        # ── 전략 검증 ──
        valid_strategies = {"rsi", "ma_cross", "bollinger"}
        if self.strategy.active not in valid_strategies:
            errors.append(
                f"[설정 오류] 알 수 없는 전략: '{self.strategy.active}'. "
                f"유효한 전략: {valid_strategies}"
            )

        # ── 전략 파라미터 검증 ──
        if self.strategy.rsi.period < 2:
            errors.append("[설정 오류] RSI period는 2 이상이어야 합니다.")
        if not (0 < self.strategy.rsi.oversold < self.strategy.rsi.overbought < 100):
            errors.append(
                "[설정 오류] RSI oversold/overbought 범위가 올바르지 않습니다. "
                "0 < oversold < overbought < 100 이어야 합니다."
            )
        if self.strategy.ma_cross.short_period >= self.strategy.ma_cross.long_period:
            errors.append(
                "[설정 오류] MA 교차 전략에서 short_period가 "
                "long_period보다 크거나 같습니다."
            )
        if self.strategy.ma_cross.ma_type not in {"SMA", "EMA"}:
            errors.append(
                f"[설정 오류] 알 수 없는 이동평균 유형: '{self.strategy.ma_cross.ma_type}'. "
                f"유효한 유형: SMA, EMA"
            )

        # ── 로깅 레벨 검증 ──
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.logging.level.upper() not in valid_levels:
            errors.append(
                f"[설정 오류] 알 수 없는 로그 레벨: '{self.logging.level}'. "
                f"유효한 레벨: {valid_levels}"
            )

        # ── 알림 채널 검증 ──
        valid_channels = {"telegram", "slack"}
        if self.notification.enabled:
            if self.notification.channel not in valid_channels:
                errors.append(
                    f"[설정 오류] 알 수 없는 알림 채널: '{self.notification.channel}'. "
                    f"유효한 채널: {valid_channels}"
                )
            # 텔레그램 채널이면 토큰/챗ID 확인
            if self.notification.channel == "telegram":
                if not self.telegram_bot_token:
                    errors.append(
                        "[알림 설정] 텔레그램 알림이 활성화되어 있지만 "
                        "TELEGRAM_BOT_TOKEN이 설정되지 않았습니다."
                    )
                if not self.telegram_chat_id:
                    errors.append(
                        "[알림 설정] 텔레그램 알림이 활성화되어 있지만 "
                        "TELEGRAM_CHAT_ID가 설정되지 않았습니다."
                    )
            # Slack 채널이면 Webhook URL 확인
            if self.notification.channel == "slack":
                if not self.slack_webhook_url:
                    errors.append(
                        "[알림 설정] Slack 알림이 활성화되어 있지만 "
                        "SLACK_WEBHOOK_URL이 설정되지 않았습니다."
                    )

        # ── 타임프레임 검증 ──
        valid_timeframes = {"minute1", "minute3", "minute5", "minute10",
                            "minute15", "minute30", "minute60", "minute240",
                            "day", "week", "month"}
        if self.trading.timeframe not in valid_timeframes:
            errors.append(
                f"[설정 오류] 알 수 없는 타임프레임: '{self.trading.timeframe}'. "
                f"유효한 값: {sorted(valid_timeframes)}"
            )

        # ── 마켓 코드 형식 검증 ──
        for market in self.trading.markets:
            if not market.startswith("KRW-"):
                errors.append(
                    f"[설정 경고] 마켓 '{market}'이 KRW 마켓이 아닙니다. "
                    f"현재 KRW 마켓만 지원합니다."
                )

        return errors

    def __repr__(self) -> str:
        """설정 요약 문자열을 반환한다."""
        return (
            f"Settings("
            f"markets={self.trading.markets}, "
            f"strategy={self.strategy.active}, "
            f"per_trade={self.investment.per_trade_amount:,}KRW, "
            f"stop_loss={self.risk.stop_loss_pct}%, "
            f"take_profit={self.risk.take_profit_pct}%"
            f")"
        )
