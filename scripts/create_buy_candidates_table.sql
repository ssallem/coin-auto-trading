-- buy_candidates 테이블 생성 스크립트
-- Python 봇의 전략 분석 결과를 저장하여 웹 대시보드에서 조회

CREATE TABLE IF NOT EXISTS buy_candidates (
  id BIGSERIAL PRIMARY KEY,
  bot_id TEXT NOT NULL DEFAULT 'main',
  market TEXT NOT NULL,
  signal TEXT NOT NULL CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
  confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
  reason TEXT,
  current_price REAL NOT NULL,
  rsi REAL,
  indicators JSONB DEFAULT '{}'::jsonb,
  analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_buy_candidates_bot_id ON buy_candidates(bot_id);
CREATE INDEX IF NOT EXISTS idx_buy_candidates_market ON buy_candidates(market);
CREATE INDEX IF NOT EXISTS idx_buy_candidates_signal ON buy_candidates(signal);
CREATE INDEX IF NOT EXISTS idx_buy_candidates_analyzed_at ON buy_candidates(analyzed_at DESC);

-- bot_id + market 복합 인덱스 (조회 성능 향상)
CREATE INDEX IF NOT EXISTS idx_buy_candidates_bot_market ON buy_candidates(bot_id, market);

-- 코멘트 추가
COMMENT ON TABLE buy_candidates IS 'Python 봇의 매 사이클 전략 분석 결과 (매수 후보 목록)';
COMMENT ON COLUMN buy_candidates.bot_id IS '봇 식별자 (기본값: main)';
COMMENT ON COLUMN buy_candidates.market IS '마켓 코드 (예: KRW-BTC)';
COMMENT ON COLUMN buy_candidates.signal IS '매매 신호 (BUY/SELL/HOLD)';
COMMENT ON COLUMN buy_candidates.confidence IS '신호 확신도 (0.0 ~ 1.0)';
COMMENT ON COLUMN buy_candidates.reason IS '신호 발생 사유';
COMMENT ON COLUMN buy_candidates.current_price IS '현재가';
COMMENT ON COLUMN buy_candidates.rsi IS 'RSI 지표 값 (nullable)';
COMMENT ON COLUMN buy_candidates.indicators IS '기술적 지표 요약 (JSON)';
COMMENT ON COLUMN buy_candidates.analyzed_at IS '분석 시각';

-- RLS (Row Level Security) 활성화 (선택 사항)
-- ALTER TABLE buy_candidates ENABLE ROW LEVEL SECURITY;

-- 읽기 권한 정책 (모든 사용자가 읽을 수 있도록)
-- CREATE POLICY "Allow read access to all users" ON buy_candidates
--   FOR SELECT USING (true);
