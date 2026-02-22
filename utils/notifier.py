"""
알림 모듈

매매 이벤트 발생 시 텔레그램 또는 Slack으로 알림을 전송한다.
표준 라이브러리의 urllib만 사용하여 외부 의존성을 최소화한다.

지원 채널:
  - Telegram (Bot API)
  - Slack (Incoming Webhook)

사용법:
    # 방법 1: 클래스 직접 사용
    notifier = Notifier(channel="telegram", telegram_bot_token="...", ...)
    notifier.send("메시지")

    # 방법 2: 통합 인터페이스 함수 (Settings 기반 자동 구성)
    send_notification("KRW-BTC 매수 체결", level="info")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional
from urllib import error, request

logger = logging.getLogger(__name__)

# 모듈 레벨 싱글턴 Notifier 인스턴스
_notifier_instance: Optional[Notifier] = None


class Notifier:
    """
    알림 전송기

    매매 체결, 손절/익절 발동, 오류 등의 이벤트를 외부 메신저로 전송한다.
    설정에서 비활성화(enabled=False) 시 모든 전송 메서드는 아무 동작 없이 False를 반환한다.

    사용법:
        notifier = Notifier(
            channel="telegram",
            telegram_bot_token="...",
            telegram_chat_id="...",
            enabled=True,
        )
        notifier.send("KRW-BTC 매수 체결: 50,000,000 KRW")
    """

    def __init__(
        self,
        channel: str = "telegram",
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
        slack_webhook_url: str = "",
        enabled: bool = False,
        events: Optional[list] = None,
    ) -> None:
        """
        Args:
            channel: 알림 채널 ("telegram" 또는 "slack")
            telegram_bot_token: 텔레그램 봇 토큰
            telegram_chat_id: 텔레그램 채팅 ID
            slack_webhook_url: Slack Incoming Webhook URL
            enabled: 알림 활성화 여부
            events: 알림을 보낼 이벤트 목록 (None이면 모든 이벤트)
        """
        self._channel = channel.lower()
        self._telegram_bot_token = telegram_bot_token
        self._telegram_chat_id = telegram_chat_id
        self._slack_webhook_url = slack_webhook_url
        self._enabled = enabled
        self._events = set(events) if events else set()

    @classmethod
    def from_settings(cls, settings) -> Notifier:
        """
        Settings 객체에서 Notifier 인스턴스를 생성한다.

        Args:
            settings: config.settings.Settings 인스턴스
        Returns:
            설정 기반 Notifier 인스턴스
        """
        return cls(
            channel=settings.notification.channel,
            telegram_bot_token=settings.telegram_bot_token,
            telegram_chat_id=settings.telegram_chat_id,
            slack_webhook_url=settings.slack_webhook_url,
            enabled=settings.notification.enabled,
            events=settings.notification.events,
        )

    # ─────────────────────────────────────
    # 공개 API: 메시지 전송
    # ─────────────────────────────────────

    def send(self, message: str, event_type: str = "") -> bool:
        """
        알림 메시지를 전송한다.

        Args:
            message: 전송할 메시지 텍스트
            event_type: 이벤트 유형 (events 필터링에 사용, 비워두면 항상 전송)
        Returns:
            전송 성공 여부
        """
        if not self._enabled:
            logger.debug(f"알림 비활성화 상태: {message[:50]}...")
            return False

        # 이벤트 필터링: events 목록이 설정되어 있고, event_type이 지정된 경우
        if event_type and self._events and event_type not in self._events:
            logger.debug(
                f"이벤트 '{event_type}'은 알림 대상이 아닙니다. "
                f"허용 이벤트: {self._events}"
            )
            return False

        try:
            if self._channel == "telegram":
                return self._send_telegram(message)
            elif self._channel == "slack":
                return self._send_slack(message)
            else:
                logger.warning(f"알 수 없는 알림 채널: {self._channel}")
                return False
        except Exception as e:
            logger.error(f"알림 전송 실패: {e}", exc_info=True)
            return False

    def send_trade_notification(
        self,
        market: str,
        side: str,
        price: float,
        amount: float,
        reason: str = "",
    ) -> bool:
        """
        매매 체결 알림을 전송한다.

        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            side: "buy" 또는 "sell"
            price: 체결 가격
            amount: 체결 금액 (KRW)
            reason: 매매 사유
        Returns:
            전송 성공 여부
        """
        action = "매수" if side == "buy" else "매도"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"[{action}] {market}\n"
            f"시간: {timestamp}\n"
            f"가격: {price:,.0f} KRW\n"
            f"금액: {amount:,.0f} KRW"
        )
        if reason:
            msg += f"\n사유: {reason}"
        return self.send(msg, event_type="order_executed")

    def send_risk_alert(
        self,
        market: str,
        alert_type: str,
        pnl_pct: float,
    ) -> bool:
        """
        리스크 알림을 전송한다 (손절/익절/트레일링 스탑).

        Args:
            market: 마켓 코드
            alert_type: "stop_loss", "take_profit", "trailing_stop"
            pnl_pct: 손익률 (%)
        Returns:
            전송 성공 여부
        """
        type_labels = {
            "stop_loss": "손절 발동",
            "take_profit": "익절 발동",
            "trailing_stop": "트레일링 스탑 발동",
        }
        # 이벤트 유형 매핑
        event_map = {
            "stop_loss": "stop_loss_triggered",
            "take_profit": "take_profit_triggered",
            "trailing_stop": "stop_loss_triggered",
        }
        label = type_labels.get(alert_type, alert_type)
        event_type = event_map.get(alert_type, alert_type)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"[{label}] {market}\n"
            f"시간: {timestamp}\n"
            f"손익률: {pnl_pct:+.2f}%"
        )
        return self.send(msg, event_type=event_type)

    def send_error_alert(self, error_message: str) -> bool:
        """
        오류 알림을 전송한다.

        Args:
            error_message: 오류 메시지 내용
        Returns:
            전송 성공 여부
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"[오류 발생]\n"
            f"시간: {timestamp}\n"
            f"{error_message}"
        )
        return self.send(msg, event_type="error")

    def send_daily_report(
        self,
        total_trades: int,
        win_count: int,
        loss_count: int,
        total_pnl: float,
        total_pnl_pct: float,
    ) -> bool:
        """
        일일 리포트를 전송한다.

        Args:
            total_trades: 총 거래 횟수
            win_count: 수익 거래 횟수
            loss_count: 손실 거래 횟수
            total_pnl: 총 손익 금액 (KRW)
            total_pnl_pct: 총 손익률 (%)
        Returns:
            전송 성공 여부
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
        msg = (
            f"[일일 리포트] {date_str}\n"
            f"총 거래: {total_trades}건\n"
            f"수익/손실: {win_count}건/{loss_count}건\n"
            f"승률: {win_rate:.1f}%\n"
            f"총 손익: {total_pnl:+,.0f} KRW ({total_pnl_pct:+.2f}%)"
        )
        return self.send(msg, event_type="daily_report")

    # ─────────────────────────────────────
    # 내부: 텔레그램 전송
    # ─────────────────────────────────────

    def _send_telegram(self, message: str) -> bool:
        """텔레그램 Bot API로 메시지를 전송한다."""
        if not self._telegram_bot_token or not self._telegram_chat_id:
            logger.warning("텔레그램 설정 누락 (bot_token 또는 chat_id)")
            return False

        url = (
            f"https://api.telegram.org/bot{self._telegram_bot_token}"
            f"/sendMessage"
        )
        payload = {
            "chat_id": self._telegram_chat_id,
            "text": message,
            "parse_mode": "HTML",
        }
        return self._post_json(url, payload)

    # ─────────────────────────────────────
    # 내부: Slack 전송
    # ─────────────────────────────────────

    def _send_slack(self, message: str) -> bool:
        """Slack Incoming Webhook으로 메시지를 전송한다."""
        if not self._slack_webhook_url:
            logger.warning("Slack Webhook URL이 설정되지 않았습니다.")
            return False

        payload = {"text": message}
        return self._post_json(self._slack_webhook_url, payload)

    # ─────────────────────────────────────
    # 내부: HTTP POST
    # ─────────────────────────────────────

    @staticmethod
    def _post_json(url: str, payload: dict) -> bool:
        """
        JSON POST 요청을 전송한다.

        표준 라이브러리 urllib만 사용하여 외부 의존성 없이 HTTP 요청을 수행한다.

        Args:
            url: 요청 URL
            payload: JSON 페이로드 딕셔너리
        Returns:
            성공 여부 (HTTP 200 응답)
        """
        try:
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.debug(f"알림 전송 성공: {url}")
                    return True
                logger.warning(f"알림 전송 응답 코드: {resp.status}")
                return False
        except error.URLError as e:
            logger.error(f"알림 전송 네트워크 오류: {e}")
            return False
        except Exception as e:
            logger.error(f"알림 전송 중 예상치 못한 오류: {e}", exc_info=True)
            return False


# ─────────────────────────────────────────────
# 통합 인터페이스 함수
# ─────────────────────────────────────────────

def init_notifier(settings) -> Notifier:
    """
    Settings 객체를 기반으로 모듈 레벨 Notifier를 초기화한다.

    애플리케이션 시작 시 한 번 호출하면 이후 send_notification()으로
    어디서든 알림을 보낼 수 있다.

    Args:
        settings: config.settings.Settings 인스턴스
    Returns:
        초기화된 Notifier 인스턴스
    """
    global _notifier_instance
    _notifier_instance = Notifier.from_settings(settings)
    logger.info(
        f"알림 모듈 초기화: channel={settings.notification.channel}, "
        f"enabled={settings.notification.enabled}"
    )
    return _notifier_instance


def get_notifier() -> Optional[Notifier]:
    """
    현재 초기화된 Notifier 인스턴스를 반환한다.

    Returns:
        Notifier 인스턴스 (초기화되지 않았으면 None)
    """
    return _notifier_instance


def send_notification(message: str, level: str = "info") -> bool:
    """
    통합 알림 전송 함수.

    모듈 레벨에서 초기화된 Notifier를 통해 알림을 전송한다.
    init_notifier()가 호출되지 않았거나, 알림이 비활성화 상태면 아무것도 하지 않는다.

    Args:
        message: 전송할 메시지
        level: 알림 수준 ("info", "warning", "error")
              - "error" 수준은 event_type="error"로 전송
              - 그 외는 event_type 없이 전송 (이벤트 필터 무시)
    Returns:
        전송 성공 여부
    """
    if _notifier_instance is None:
        logger.debug("알림 모듈이 초기화되지 않았습니다.")
        return False

    # level에 따라 이벤트 유형 결정
    event_type = ""
    if level.lower() == "error":
        event_type = "error"

    return _notifier_instance.send(message, event_type=event_type)
