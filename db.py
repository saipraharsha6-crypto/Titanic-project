
import psycopg2
import pandas as pd

def get_connection():
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="titanic_db",
            user="postgres",
            password="Postgres@123",
            port="5432"
        )
        return connection

    except Exception as e:
        print("Database connection error:", e)
        return None


def get_passengers():
    connection = get_connection()

    if connection is None:
        return pd.DataFrame()

    query = """
        SELECT *
        FROM titanic_project.passengers;
    """

    df = pd.read_sql_query(query, connection)
    connection.close()

    return df
if __name__ == "__main__":
    conn = get_connection()

    if conn:
        print("Database connected successfully!")

        df = get_passengers()
        print("Total passengers:", len(df))

        conn.close()
