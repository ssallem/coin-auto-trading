"""
매매 전략 기본 인터페이스 (추상 클래스)

모든 매매 전략은 이 추상 클래스를 상속하여 구현한다.
Strategy 패턴을 적용하여 전략을 교체 가능하게 설계한다.

전략 신호:
  - BUY: 매수 신호
  - SELL: 매도 신호
  - HOLD: 관망 (아무것도 하지 않음)

SignalResult:
  - signal: 매매 방향 (Signal enum)
  - market: 마켓 코드 (예: "KRW-BTC")
  - confidence: 신호 확신도 (0.0 ~ 1.0)
  - reason: 사람이 읽을 수 있는 판단 근거
  - metadata: 전략별 추가 데이터 (지표 값 등)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class Signal(Enum):
    """매매 신호"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalResult:
    """
    전략이 반환하는 매매 신호 결과

    Attributes:
        signal: 매매 신호 (BUY, SELL, HOLD)
        market: 마켓 코드
        confidence: 신호 확신도 (0.0 ~ 1.0, 높을수록 강한 신호)
        reason: 신호 발생 사유 (로깅/분석용)
        metadata: 추가 메타데이터 (전략별 상세 정보)
    """
    signal: Signal
    market: str
    confidence: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    매매 전략 추상 기본 클래스

    모든 전략은 이 클래스를 상속하고, name 프로퍼티와 analyze() 메서드를 구현해야 한다.
    TradingEngine은 이 인터페이스를 통해 전략과 상호작용한다.

    필수 구현:
        - name (property): 전략의 고유 이름
        - analyze(): 시세 데이터를 분석하여 매매 신호를 반환

    선택적 오버라이드:
        - on_position_opened(): 포지션 진입 시 호출되는 훅
        - on_position_closed(): 포지션 종료 시 호출되는 훅
        - reset(): 전략 내부 상태 초기화 (백테스트 시작 시 호출)

    사용 예시:
        class MyStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "my_strategy"

            def analyze(self, market, df, current_price) -> SignalResult:
                # 지표 계산 및 매매 판단 로직
                ...
                return SignalResult(signal=Signal.BUY, market=market, ...)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """전략 이름을 반환한다. 로깅과 식별에 사용된다."""
        ...

    @abstractmethod
    def analyze(
        self,
        market: str,
        df: pd.DataFrame,
        current_price: float,
    ) -> SignalResult:
        """
        시세 데이터를 분석하여 매매 신호를 생성한다.

        이 메서드는 TradingEngine의 매 사이클마다 호출된다.
        전달받은 DataFrame에는 OHLCV 데이터가 포함되어 있다.

        구현 시 주의사항:
          - 충분한 데이터가 없을 경우 Signal.HOLD를 반환해야 한다.
          - confidence는 0.0 ~ 1.0 범위를 유지해야 한다.
          - reason에는 사람이 읽을 수 있는 판단 근거를 포함한다.

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            df: OHLCV DataFrame (충분한 과거 데이터 포함)
            current_price: 현재가
        Returns:
            SignalResult 객체
        """
        ...

    def on_position_opened(self, market: str, price: float, amount: float) -> None:
        """
        포지션이 열릴 때 호출되는 훅.
        전략이 내부 상태를 관리해야 할 경우 서브클래스에서 오버라이드한다.

        Args:
            market: 마켓 코드
            price: 체결 가격
            amount: 체결 금액/수량
        """
        logger.debug(
            "[%s] %s 포지션 진입: price=%.2f, amount=%.4f",
            self.name,
            market,
            price,
            amount,
        )

    def on_position_closed(self, market: str, profit_pct: float) -> None:
        """
        포지션이 닫힐 때 호출되는 훅.
        전략이 내부 상태를 관리해야 할 경우 서브클래스에서 오버라이드한다.

        Args:
            market: 마켓 코드
            profit_pct: 수익률 (%)
        """
        logger.debug(
            "[%s] %s 포지션 종료: profit=%.2f%%",
            self.name,
            market,
            profit_pct,
        )

    def reset(self) -> None:
        """
        전략 내부 상태를 초기화한다.
        백테스트 시작 시 호출되어 이전 상태를 정리한다.
        """
        logger.debug("[%s] 전략 상태 초기화", self.name)
