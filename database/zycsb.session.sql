-- @block
CREATE TABLE Users(
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    bio TEXT,
    country VARCHAR(2)
);

-- @block
INSERT INTO Users(email, bio, country)
VALUES 
(
    'cate@example.com', 
    'I like photography .', 
    'US'
),
(
    'jade@example.com', 
    'I like to snowboard.', 
    'CN'
);

-- @block
SELECT email, id, country FROM Users

WHERE country = 'US'
AND id > 0
AND email LIKE '%@example.com'

ORDER BY id DESC
LIMIT 2;

-- @block
CREATE INDEX email_index ON Users(email);

-- @block
CREATE TABLE Rooms(
    id INT AUTO_INCREMENT,
    street VARCHAR(255),
    owner_id INT NOT NULL,
    PRIMARY KEY(id),
    FOREIGN KEY(owner_id) REFERENCES Users(id)
);


-- @block
INSERT INTO Rooms(owner_id, street)
VALUES
(1, 'fanqiejidanlu'),
(1, 'qingjiaorousilu'),
(1, 'qingjiaodaxiaolu');

-- @block
SELECT * FROM Users
INNER JOIN Rooms 
ON Users.id = Rooms.owner_id