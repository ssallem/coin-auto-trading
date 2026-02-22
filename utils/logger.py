"""
로깅 모듈

애플리케이션 전역 로거를 설정한다.
콘솔과 파일에 동시에 로그를 출력하며,
파일은 RotatingFileHandler로 크기 제한을 둔다.

사용법:
    from utils.logger import setup_logger, get_logger

    # 앱 시작 시 한 번 호출
    setup_logger(level="INFO", log_file="logs/trading.log")

    # 각 모듈에서 로거 획득
    logger = get_logger(__name__)
    logger.info("매매 시작")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# 모듈 레벨 변수: setup_logger 호출 여부 추적
_initialized: bool = False


def setup_logger(
    level: str = "INFO",
    log_file: str = "logs/trading.log",
    max_size_mb: int = 10,
    backup_count: int = 5,
) -> logging.Logger:
    """
    루트 로거를 설정한다.

    콘솔 핸들러와 RotatingFileHandler를 등록하여
    터미널과 파일에 동시에 로그를 기록한다.

    Args:
        level: 로그 레벨 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: 로그 파일 경로 (프로젝트 루트 기준 상대 경로 가능)
        max_size_mb: 로그 파일 최대 크기 (MB), 초과 시 로테이션
        backup_count: 로그 파일 백업 개수
    Returns:
        설정된 루트 로거
    """
    global _initialized

    # 로그 디렉토리 자동 생성
    log_path = Path(log_file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[경고] 로그 디렉토리 생성 실패: {e}", file=sys.stderr)

    # 로그 포맷 정의
    fmt = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # 기존 핸들러 제거 (중복 방지 - 재초기화 시에도 안전)
    root_logger.handlers.clear()

    # ── 콘솔 핸들러 ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # ── 파일 핸들러 (RotatingFileHandler) ──
    try:
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except OSError as e:
        # 파일 핸들러 생성 실패 시 콘솔만 사용
        root_logger.warning(f"로그 파일 핸들러 생성 실패 (콘솔만 사용): {e}")

    # ── 외부 라이브러리 로그 레벨 조정 (지나치게 상세한 로그 억제) ──
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    _initialized = True
    root_logger.info(
        f"로거 초기화 완료: level={level}, file={log_file}, "
        f"max_size={max_size_mb}MB, backup_count={backup_count}"
    )
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거를 반환한다.

    setup_logger()가 아직 호출되지 않은 경우에도 안전하게 동작한다.
    이 경우 기본 WARNING 레벨의 로거가 반환된다.

    Args:
        name: 로거 이름 (일반적으로 __name__ 전달)
    Returns:
        해당 이름의 로거 인스턴스

    사용법:
        logger = get_logger(__name__)
        logger.info("거래 시작")
        logger.error("API 호출 실패", exc_info=True)
    """
    if not _initialized:
        # setup_logger가 아직 호출되지 않은 경우 기본 핸들러 설정
        # (라이브러리로 단독 사용될 때를 위한 안전장치)
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.WARNING,
                format="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    return logging.getLogger(name)
