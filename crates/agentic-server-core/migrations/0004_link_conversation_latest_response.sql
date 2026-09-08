ALTER TABLE conversations
    ADD COLUMN latest_response_id TEXT REFERENCES responses(id) ON DELETE SET NULL;
