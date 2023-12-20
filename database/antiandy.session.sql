-- @block
CREATE TABLE IF NOT EXISTS question_bank (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT,
    embedding BLOB,
    paper VARCHAR(255),
    page INT,
    q_number INT
);

-- @block