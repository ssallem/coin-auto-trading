"""
Supabase bot_config 테이블 업데이트 스크립트

config.yaml의 설정을 읽어서 Supabase bot_config 테이블(id=1)에 upsert한다.

사용법:
    python scripts/update_supabase_config.py
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import requests
from dotenv import load_dotenv


def load_config_yaml():
    """config.yaml을 로드하여 딕셔너리로 반환"""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("config.yaml이 올바른 형식이 아닙니다.")

    return data


def convert_to_supabase_format(config):
    """
    config.yaml 형식을 Supabase 형식으로 변환
    (trailing_stop 플랫 구조 → 중첩 구조)
    """
    result = dict(config)

    # risk.trailing_stop_enabled, trailing_stop_pct를 중첩 구조로 변환
    if "risk" in result:
        risk = dict(result["risk"])

        # trailing_stop 중첩 구조 생성
        trailing_enabled = risk.pop("trailing_stop_enabled", False)
        trailing_pct = risk.pop("trailing_stop_pct", 2.0)

        risk["trailing_stop"] = {
            "enabled": trailing_enabled,
            "pct": trailing_pct
        }

        result["risk"] = risk

    return result


def update_supabase_config(config_data):
    """Supabase bot_config 테이블 업데이트"""
    # 환경변수 로드
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase 환경변수가 설정되지 않았습니다.\n"
            ".env 파일에 SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY를 설정하세요."
        )

    # Supabase 형식으로 변환
    supabase_config = convert_to_supabase_format(config_data)

    # id 필드 추가 (upsert를 위해)
    supabase_config["id"] = 1

    # Supabase REST API 호출 (upsert)
    url = f"{supabase_url}/rest/v1/bot_config"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    try:
        response = requests.post(
            url,
            json=supabase_config,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Supabase 업데이트 실패: {e}")


def print_config_summary(config):
    """설정 요약 출력"""
    print("\n" + "=" * 60)
    print("  현재 config.yaml 설정 요약")
    print("=" * 60)

    # 거래 설정
    trading = config.get("trading", {})
    print("\n[거래 설정]")
    markets = trading.get("markets", [])
    print(f"  마켓 수:       {len(markets)}개")
    if len(markets) <= 5:
        print(f"  마켓 목록:     {', '.join(markets)}")
    else:
        print(f"  마켓 목록:     {', '.join(markets[:3])}, ... (+{len(markets)-3}개)")
    print(f"  폴링 간격:     {trading.get('poll_interval', 0)}초")
    print(f"  타임프레임:    {trading.get('timeframe', '')}")
    print(f"  캔들 수:       {trading.get('candle_count', 0)}개")

    # 투자 설정
    investment = config.get("investment", {})
    print("\n[투자 설정]")
    print(f"  총 투자금:     {investment.get('max_total_investment', 0):,} KRW")
    print(f"  1회 매수금:    {investment.get('per_trade_amount', 0):,} KRW")
    print(f"  최소 주문금:   {investment.get('min_order_amount', 0):,} KRW")

    # 리스크 설정
    risk = config.get("risk", {})
    print("\n[리스크 설정]")
    print(f"  손절:          {risk.get('stop_loss_pct', 0)}%")
    print(f"  익절:          {risk.get('take_profit_pct', 0)}%")
    print(f"  최대 포지션:   {risk.get('max_positions', 0)}개")
    print(f"  일일 최대손실: {risk.get('max_daily_loss', 0):,} KRW")
    print(f"  트레일링 스탑: {'활성화' if risk.get('trailing_stop_enabled', False) else '비활성화'}")
    if risk.get('trailing_stop_enabled', False):
        print(f"  트레일링 비율: {risk.get('trailing_stop_pct', 0)}%")

    # 전략 설정
    strategy = config.get("strategy", {})
    print("\n[전략 설정]")
    print(f"  활성 전략:     {strategy.get('active', '')}")

    print("\n" + "=" * 60)


def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("  Supabase bot_config 업데이트 스크립트")
    print("=" * 60)

    try:
        # 1. config.yaml 로드
        print("\n[1/3] config.yaml 로드 중...")
        config = load_config_yaml()
        print("  -> 성공: 설정 파일을 로드했습니다.")

        # 2. 설정 요약 출력
        print_config_summary(config)

        # 3. 사용자 확인
        print("\n[2/3] 위 설정을 Supabase에 업데이트하시겠습니까?")
        answer = input("  계속하려면 'y' 또는 'yes'를 입력하세요: ").strip().lower()

        if answer not in ["y", "yes"]:
            print("\n  -> 취소됨: 사용자가 업데이트를 취소했습니다.")
            return

        # 4. Supabase 업데이트
        print("\n[3/3] Supabase bot_config 업데이트 중...")
        update_supabase_config(config)
        print("  -> 성공: Supabase bot_config 테이블(id=1)이 업데이트되었습니다.")

        print("\n" + "=" * 60)
        print("  완료: 설정이 성공적으로 동기화되었습니다!")
        print("=" * 60 + "\n")

    except FileNotFoundError as e:
        print(f"\n[오류] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[오류] {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n[오류] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[오류] 예기치 않은 오류: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
