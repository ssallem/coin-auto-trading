-- coin_locks 테이블: 웹에서 코인 잠금 시 자동 매도를 방지한다.
-- 잠금된 코인은 봇의 리스크 매니저(손절/익절/트레일링)에 의한 매도가 차단된다.

CREATE TABLE IF NOT EXISTS coin_locks (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    market      text UNIQUE NOT NULL,        -- 예: "KRW-BTC"
    is_locked   boolean DEFAULT true,
    locked_at   timestamptz DEFAULT now(),
    updated_at  timestamptz DEFAULT now()
);

-- updated_at 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_coin_locks_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_coin_locks_updated_at
    BEFORE UPDATE ON coin_locks
    FOR EACH ROW
    EXECUTE FUNCTION update_coin_locks_updated_at();

-- RLS 활성화
ALTER TABLE coin_locks ENABLE ROW LEVEL SECURITY;

-- anon/authenticated 모두 SELECT, INSERT, UPDATE 허용
CREATE POLICY "coin_locks_select_policy"
    ON coin_locks FOR SELECT
    TO anon, authenticated
    USING (true);

CREATE POLICY "coin_locks_insert_policy"
    ON coin_locks FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);

CREATE POLICY "coin_locks_update_policy"
    ON coin_locks FOR UPDATE
    TO anon, authenticated
    USING (true)
    WITH CHECK (true);
