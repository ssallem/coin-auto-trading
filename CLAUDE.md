# CoinAutoTrading Bot

Upbit 거래소 자동매매 프로그램 (Python)

## 프로젝트 구조

```
CoinAutoTrading/
├── main.py                 # 메인 엔트리포인트
├── config/
│   ├── config.yaml         # 전체 설정 (거래/리스크/전략/로깅)
│   └── settings.py         # 설정 로더 (싱글턴)
├── api/
│   └── upbit_client.py     # Upbit API 래퍼 (pyupbit)
├── data/
│   ├── collector.py        # 시세 데이터 수집 (REST + WebSocket)
│   └── indicators.py       # 기술적 분석 지표 (RSI, MA, BB, MACD)
├── strategies/
│   ├── base_strategy.py    # 전략 추상 클래스 (Strategy 패턴)
│   ├── rsi_strategy.py     # RSI 전략
│   ├── ma_cross_strategy.py # 이동평균 교차 전략
│   └── bollinger_strategy.py # 볼린저 밴드 전략
├── trading/
│   ├── engine.py           # 메인 트레이딩 엔진 (오케스트레이터)
│   ├── order_manager.py    # 주문 관리
│   └── risk_manager.py     # 리스크 관리 (손절/익절/트레일링)
├── backtesting/
│   └── engine.py           # 백테스트 엔진
├── utils/
│   ├── logger.py           # 로깅 설정
│   └── notifier.py         # 알림 (Telegram/Slack)
└── tests/                  # 테스트
```

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# .env 파일 설정 (.env.example 참고)
cp .env.example .env

# 설정 검증
python main.py check

# 백테스트
python main.py backtest

# 실시간 자동매매
python main.py trade
```

## 핵심 규칙

- API 키는 반드시 .env 파일에 저장 (절대 코드에 하드코딩 금지)
- 모든 거래는 RiskManager를 거쳐 안전성 검증 후 실행
- 새 전략 추가 시 BaseStrategy를 상속하여 analyze() 구현
- config.yaml의 설정값으로 전략 파라미터 변경 가능 (코드 수정 불필요)
