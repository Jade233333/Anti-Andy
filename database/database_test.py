import mysql.connector

# create a connection to the database
cnx = mysql.connector.connect(
    host='localhost',
    user='root',
    password='hello',
    database='anti_andy'
)

    
# create a cursor object
cursor = cnx.cursor()

# data to insert
data = {
    'text': 'Example text',
    'embedding': 'Example embedding',
    'paper': 'Example paper',
    'page': 1,
    'q_number': 1
}

# prepare insert statement
add_data = ("INSERT INTO question_bank"
            "(text, embedding, paper, page, q_number) "
            "VALUES (%(text)s, %(embedding)s, %(paper)s, %(page)s, %(q_number)s)")

# execute the statement
cursor.execute(add_data, data)

# commit the changes
cnx.commit()

# close the cursor and connection
cursor.close()
cnx.close()
