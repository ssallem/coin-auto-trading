"""
Supabase REST API를 통해 bot_config를 로드하는 모듈.
"""
import os
from typing import Any, Dict, Optional
import requests


def load_config_from_supabase() -> Optional[Dict[str, Any]]:
    """
    Supabase bot_config 테이블에서 id=1 row를 읽어 딕셔너리로 반환.
    환경변수 SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY 필요.
    실패 시 None 반환 (호출 측에서 yaml fallback 처리).
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        return None

    try:
        resp = requests.get(
            f"{url}/rest/v1/bot_config",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            params={"id": "eq.1", "select": "*"},
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        return rows[0]
    except Exception as e:
        print(f"[경고] Supabase 설정 로드 실패: {e}", flush=True)
        return None
