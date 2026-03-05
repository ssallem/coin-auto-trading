# 매수 후보 동기화 기능 (buy_candidates)

## 개요

Python 봇의 전략 분석 결과를 Supabase `buy_candidates` 테이블에 동기화하여, 웹 대시보드에서 실시간으로 매수 후보 목록을 조회할 수 있습니다.

## 주요 기능

- **실시간 전략 분석 결과 동기화**: 매 사이클마다 모든 마켓의 분석 결과를 Supabase에 업로드
- **기술적 지표 포함**: RSI, 이동평균, 볼린저 밴드, MACD 등 주요 지표 값 포함
- **Rate Limit 해결**: 웹 대시보드가 Upbit API를 직접 호출하지 않고 봇의 분석 결과를 재사용

## 데이터베이스 설정

### 1. Supabase 테이블 생성

Supabase 대시보드에서 SQL Editor를 열고 `scripts/create_buy_candidates_table.sql` 파일의 내용을 실행하세요.

```sql
-- scripts/create_buy_candidates_table.sql 내용을 복사하여 실행
```

### 2. 테이블 스키마

```
buy_candidates
├── id (BIGSERIAL PRIMARY KEY)
├── bot_id (TEXT) - 봇 식별자 (기본값: "main")
├── market (TEXT) - 마켓 코드 (예: "KRW-BTC")
├── signal (TEXT) - 매매 신호 (BUY/SELL/HOLD)
├── confidence (REAL) - 신호 확신도 (0.0 ~ 1.0)
├── reason (TEXT) - 신호 발생 사유
├── current_price (REAL) - 현재가
├── rsi (REAL) - RSI 지표 값 (nullable)
├── indicators (JSONB) - 기술적 지표 요약
├── analyzed_at (TIMESTAMPTZ) - 분석 시각
└── created_at (TIMESTAMPTZ) - 레코드 생성 시각
```

## 코드 변경 사항

### 1. sync/supabase_sync.py

새로운 메서드 추가:

```python
def push_buy_candidates(self, candidates: List[Dict[str, Any]]) -> bool:
    """
    buy_candidates 테이블에 전략 분석 결과를 업로드한다.
    매 사이클마다 기존 bot_id 데이터를 삭제하고 새로운 분석 결과로 교체한다.
    """
```

**동작 방식:**
- 매 사이클마다 기존 `bot_id='main'` 데이터를 삭제
- 새로운 분석 결과를 일괄 INSERT
- 실패 시에도 봇 동작에 영향 없음 (경고 로그만 출력)

### 2. trading/engine.py

**주요 변경:**
- `_execute_cycle()`: 매 사이클 시작 시 `_buy_candidates` 리스트 초기화
- `_process_market()`: 전략 분석 후 `_collect_buy_candidate()` 호출하여 결과 수집
- `_collect_buy_candidate()`: SignalResult와 DataFrame에서 지표 값 추출
- `_sync_buy_candidates()`: 사이클 종료 시 Supabase에 일괄 업로드

**수집되는 데이터:**
- market: 마켓 코드
- signal: BUY/SELL/HOLD
- confidence: 확신도 (0.0 ~ 1.0)
- reason: 신호 사유
- current_price: 현재가
- rsi: RSI 값 (있는 경우)
- indicators: 주요 지표 (rsi, ma_short, ma_long, bb_upper, bb_lower, macd, signal)

## 웹 대시보드 연동

### 조회 쿼리 예시

```javascript
// 최신 매수 후보 조회
const { data, error } = await supabase
  .from('buy_candidates')
  .select('*')
  .eq('bot_id', 'main')
  .order('confidence', { ascending: false });

// BUY 신호만 필터링
const { data, error } = await supabase
  .from('buy_candidates')
  .select('*')
  .eq('bot_id', 'main')
  .eq('signal', 'BUY')
  .gte('confidence', 0.7)  // 확신도 70% 이상
  .order('confidence', { ascending: false });

// 특정 마켓 조회
const { data, error } = await supabase
  .from('buy_candidates')
  .select('*')
  .eq('bot_id', 'main')
  .eq('market', 'KRW-BTC')
  .single();
```

### 실시간 구독 (Realtime)

```javascript
// 매수 후보 변경사항 실시간 구독
const subscription = supabase
  .channel('buy_candidates_changes')
  .on(
    'postgres_changes',
    {
      event: '*',
      schema: 'public',
      table: 'buy_candidates',
      filter: 'bot_id=eq.main'
    },
    (payload) => {
      console.log('매수 후보 업데이트:', payload);
      // UI 갱신 로직
    }
  )
  .subscribe();
```

## 로그 확인

봇 실행 시 다음과 같은 로그가 출력됩니다:

```
[DEBUG] [KRW-BTC] 매수 후보 수집: BUY (확신도: 0.85)
[DEBUG] [KRW-ETH] 매수 후보 수집: HOLD (확신도: 0.50)
[INFO] 매수 후보 5건 업로드 완료 (bot_id=main)
```

## 트러블슈팅

### 1. "매수 후보 업로드 실패" 경고

**원인:**
- SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY 환경변수 미설정
- Supabase 테이블 미생성
- 네트워크 오류

**해결:**
1. `.env` 파일에 Supabase 자격 증명 확인
2. `create_buy_candidates_table.sql` 실행 확인
3. Supabase 대시보드에서 테이블 존재 확인

### 2. 테이블에 데이터가 없음

**원인:**
- 봇이 아직 한 사이클도 실행하지 않음
- SupabaseSync가 비활성 상태

**해결:**
1. 봇 로그에서 "SupabaseSync 초기화 완료 (활성)" 메시지 확인
2. 최소 1회 사이클 완료 대기 (설정된 interval 시간)

### 3. 데이터가 업데이트되지 않음

**원인:**
- 봇이 중지됨
- 매 사이클마다 DELETE → INSERT 수행하므로 데이터가 없다면 봇이 동작하지 않는 것

**해결:**
1. `python main.py trade` 명령으로 봇 실행 상태 확인
2. 로그 파일 확인: `logs/app.log`

## 성능 고려사항

- **Rate Limit 해결**: 웹 대시보드는 Upbit API를 호출하지 않으므로 Rate Limit 문제 없음
- **데이터 신선도**: 봇의 사이클 간격(기본 60초)에 따라 최대 60초 지연 가능
- **테이블 크기**: 매 사이클마다 전체 교체하므로 레코드 수는 `마켓 수 = buy_candidates 레코드 수`로 일정
- **쿼리 성능**: bot_id, market 인덱스가 생성되어 있어 조회 성능 우수

## 향후 개선 사항

- [ ] 히스토리 테이블 추가 (과거 분석 결과 보관)
- [ ] 알림 기능 (확신도 높은 BUY 신호 발생 시)
- [ ] 멀티봇 지원 (bot_id 별 독립적 관리)
- [ ] 백테스트 결과 저장
