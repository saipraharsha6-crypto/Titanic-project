
import streamlit as st
import pandas as pd
import plotly.express as px
from db import get_connection

# Page configuration
st.set_page_config(
    page_title="Titanic Dataset Analysis",
    page_icon="🚢",
    layout="wide"
)

# Load data from PostgreSQL
@st.cache_data
def load_data():
    conn = get_connection()

    if conn is None:
        return pd.DataFrame()

    query = """
        SELECT
            p.passenger_id,
            p.name,
            p.sex,
            p.age,
            s.pclass,
            s.survived,
            s.sibsp,
            s.parch
        FROM titanic_project.passengers p
        JOIN titanic_project.survival_details s
        ON p.passenger_id = s.passenger_id
        ORDER BY p.passenger_id;
    """

    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


# Main title
st.title("🚢 Titanic Dataset Analysis Dashboard")
st.write("Explore Titanic passenger data using PostgreSQL and Python.")

df = load_data()

if df.empty:
    st.error("No data found. Please check your PostgreSQL connection.")

else:
    # Data preparation
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["survived"] = pd.to_numeric(df["survived"], errors="coerce")
    df["pclass"] = pd.to_numeric(df["pclass"], errors="coerce")

    st.success("Successfully connected to PostgreSQL!")

    # Sidebar menu
    st.sidebar.title("Dashboard Menu")

    page = st.sidebar.radio(
        "Select Page",
        [
            "Dashboard",
            "Data Explorer",
            "Survival Analysis",
            "SQL Lab",
            "Interactive Test"
        ]
    )

    # =====================================
    # 1. DASHBOARD
    # =====================================
    if page == "Dashboard":

        st.header("📊 Passenger Overview")

        col1, col2, col3, col4 = st.columns(4)

        total = len(df)
        survived = int((df["survived"] == 1).sum())
        not_survived = int((df["survived"] == 0).sum())
        survival_rate = survived / total * 100 if total else 0

        col1.metric("Total Passengers", total)
        col2.metric("Survived", survived)
        col3.metric("Did Not Survive", not_survived)
        col4.metric("Survival Rate", f"{survival_rate:.1f}%")

        st.divider()

        # Survival donut chart
        st.subheader("🍩 Passenger Survival Distribution")

        survival_counts = (
            df["survived"]
            .map({0: "Did Not Survive", 1: "Survived"})
            .value_counts()
            .reindex(["Survived", "Did Not Survive"], fill_value=0)
        )

        fig1 = px.pie(
            names=survival_counts.index,
            values=survival_counts.values,
            hole=0.55,
            title="Survived vs Did Not Survive"
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Gender donut chart
        st.subheader("🍩 Gender Distribution")

        gender_counts = df["sex"].value_counts()

        fig2 = px.pie(
            names=gender_counts.index,
            values=gender_counts.values,
            hole=0.55,
            title="Male and Female Passengers"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Survival by passenger class
        st.subheader("📈 Survival Rate by Passenger Class")

        class_rate = (
            df.groupby("pclass")["survived"]
            .mean()
            .mul(100)
            .reset_index(name="survival_rate")
        )

        fig3 = px.line(
            class_rate,
            x="pclass",
            y="survival_rate",
            markers=True,
            title="Survival Rate by Passenger Class",
            labels={
                "pclass": "Passenger Class",
                "survival_rate": "Survival Rate (%)"
            }
        )

        st.plotly_chart(fig3, use_container_width=True)

        # Survival by age
        st.subheader("📈 Survival Rate by Age")

        age_data = df.dropna(subset=["age", "survived"]).copy()

        age_data["age_group"] = (
            (age_data["age"] // 10) * 10
        ).astype(int)

        age_rate = (
            age_data.groupby("age_group")["survived"]
            .mean()
            .mul(100)
            .reset_index(name="survival_rate")
        )

        fig4 = px.line(
            age_rate,
            x="age_group",
            y="survival_rate",
            markers=True,
            title="Survival Rate by Age Group",
            labels={
                "age_group": "Age Group",
                "survival_rate": "Survival Rate (%)"
            }
        )

        st.plotly_chart(fig4, use_container_width=True)

        # Missing data
        st.subheader("Missing Data Summary")

        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ["Column", "Missing Values"]

        st.dataframe(missing_data, use_container_width=True)

    # =====================================
    # 2. DATA EXPLORER
    # =====================================
    elif page == "Data Explorer":

        st.header("🔍 Titanic Passenger Data Explorer")
        st.write("Filter and explore passenger records.")

        gender_options = ["All"] + sorted(
            df["sex"].dropna().unique().tolist()
        )

        selected_gender = st.selectbox(
            "Select Gender",
            gender_options
        )

        class_options = ["All"] + sorted(
            df["pclass"].dropna().unique().tolist()
        )

        selected_class = st.selectbox(
            "Select Passenger Class",
            class_options
        )

        survival_options = ["All", "Survived", "Did Not Survive"]

        selected_survival = st.selectbox(
            "Select Survival Status",
            survival_options
        )

        filtered_df = df.copy()

        if selected_gender != "All":
            filtered_df = filtered_df[
                filtered_df["sex"] == selected_gender
            ]

        if selected_class != "All":
            filtered_df = filtered_df[
                filtered_df["pclass"] == selected_class
            ]

        if selected_survival == "Survived":
            filtered_df = filtered_df[
                filtered_df["survived"] == 1
            ]

        elif selected_survival == "Did Not Survive":
            filtered_df = filtered_df[
                filtered_df["survived"] == 0
            ]

        st.metric("Filtered Passengers", len(filtered_df))

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        # Download CSV
        csv = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Passenger Data",
            data=csv,
            file_name="titanic_passengers.csv",
            mime="text/csv"
        )

    # =====================================
    # 3. SURVIVAL ANALYSIS
    # =====================================
    elif page == "Survival Analysis":

        st.header("📈 Detailed Survival Analysis")

        # Survival by gender
        gender_survival = pd.crosstab(
            df["sex"],
            df["survived"].map({
                0: "Did Not Survive",
                1: "Survived"
            })
        )

        fig5 = px.bar(
            gender_survival,
            barmode="group",
            title="Survival by Gender",
            labels={
                "value": "Number of Passengers",
                "sex": "Gender"
            }
        )

        st.plotly_chart(fig5, use_container_width=True)

        # Survival by passenger class
        class_survival = pd.crosstab(
            df["pclass"],
            df["survived"].map({
                0: "Did Not Survive",
                1: "Survived"
            })
        )

        fig6 = px.bar(
            class_survival,
            barmode="group",
            title="Survival by Passenger Class",
            labels={
                "value": "Number of Passengers",
                "pclass": "Passenger Class"
            }
        )

        st.plotly_chart(fig6, use_container_width=True)

        # Age distribution
        fig7 = px.histogram(
            df,
            x="age",
            color="sex",
            title="Passenger Age Distribution",
            nbins=20
        )

        st.plotly_chart(fig7, use_container_width=True)

        # Summary
        st.subheader("Survival Summary")

        st.write("Total passengers:", len(df))
        st.write("Survived:", int((df["survived"] == 1).sum()))
        st.write(
            "Did not survive:",
            int((df["survived"] == 0).sum())
        )

    # =====================================
    # 4. SQL LAB
    # =====================================
    elif page == "SQL Lab":

        st.header("🧪 SQL Lab")
        st.write("Practice SQL queries using PostgreSQL.")

        query_options = {
            "Show all passengers":
                "SELECT * FROM titanic_project.passengers LIMIT 20",

            "Total passengers":
                "SELECT COUNT(*) AS total_passengers FROM titanic_project.passengers",

            "Survival count":
                """SELECT survived, COUNT(*) AS total
                   FROM titanic_project.survival_details
                   GROUP BY survived""",

            "Survival by gender":
                """SELECT p.sex, s.survived, COUNT(*) AS total
                   FROM titanic_project.passengers p
                   JOIN titanic_project.survival_details s
                   ON p.passenger_id = s.passenger_id
                   GROUP BY p.sex, s.survived
                   ORDER BY p.sex""",

            "Average passenger age":
                "SELECT AVG(age) AS average_age FROM titanic_project.passengers",

            "Passenger class count":
                """SELECT pclass, COUNT(*) AS total
                   FROM titanic_project.survival_details
                   GROUP BY pclass ORDER BY pclass"""
        }

        selected_query = st.selectbox(
            "Choose an SQL example",
            list(query_options.keys())
        )

        sql_query = st.text_area(
            "Write your SQL query",
            value=query_options[selected_query],
            height=150
        )

        if st.button("Run SQL Query"):

            cleaned_query = sql_query.strip().lower().rstrip(";")

            if (
                not cleaned_query.startswith(("select", "with"))
                or ";" in cleaned_query
            ):
                st.error("Only one read-only SELECT query is allowed.")

            else:
                conn = get_connection()

                if conn is not None:
                    try:
                        conn.set_session(readonly=True)

                        result = pd.read_sql_query(
                            sql_query,
                            conn
                        )

                        st.success("Query executed successfully!")

                        st.dataframe(
                            result,
                            use_container_width=True,
                            hide_index=True
                        )

                    except Exception as e:
                        st.error(f"SQL Error: {e}")

                    finally:
                        conn.close()

    # =====================================
    # 5. INTERACTIVE TEST
    # =====================================
    elif page == "Interactive Test":

        st.header("📝 Titanic Interactive Test")
        st.write("Answer these questions to test your knowledge.")

        questions = [
            {
                "question": "1. How many passengers are in the Titanic dataset?",
                "options": ["500", "891", "1000", "1200"],
                "answer": "891"
            },
            {
                "question": "2. Which Python library is used for data analysis?",
                "options": ["Pandas", "HTML", "CSS", "Java"],
                "answer": "Pandas"
            },
            {
                "question": "3. Which database is used in this project?",
                "options": ["MySQL", "MongoDB", "PostgreSQL", "Oracle"],
                "answer": "PostgreSQL"
            },
            {
                "question": "4. Which chart is circular?",
                "options": ["Line chart", "Donut chart", "Bar chart", "Histogram"],
                "answer": "Donut chart"
            },
            {
                "question": "5. Which column indicates survival?",
                "options": ["age", "sex", "survived", "name"],
                "answer": "survived"
            },
            {
                "question": "6. Which library creates interactive charts?",
                "options": ["Plotly", "OS", "Math", "Random"],
                "answer": "Plotly"
            },
            {
                "question": "7. What does NULL represent in a database?",
                "options": ["Zero", "Missing value", "One", "Text"],
                "answer": "Missing value"
            },
            {
                "question": "8. Which command retrieves data from SQL?",
                "options": ["INSERT", "DELETE", "SELECT", "UPDATE"],
                "answer": "SELECT"
            },
            {
                "question": "9. What is the total number of columns in the original Titanic CSV?",
                "options": ["5", "8", "12", "20"],
                "answer": "12"
            },
            {
                "question": "10. Which tool displays the project dashboard?",
                "options": ["Streamlit", "Notepad", "Paint", "Calculator"],
                "answer": "Streamlit"
            }
        ]

        with st.form("titanic_quiz"):

            user_answers = []

            for q in questions:
                answer = st.radio(
                    q["question"],
                    q["options"],
                    key=q["question"]
                )
                user_answers.append(answer)

            submitted = st.form_submit_button("Submit Test")

        if submitted:

            score = 0

            st.divider()
            st.subheader("Test Results")

            for i, q in enumerate(questions):

                if user_answers[i] == q["answer"]:
                    score += 1
                    st.success(
                        f"Question {i + 1}: Correct!"
                    )

                else:
                    st.error(
                        f"Question {i + 1}: Correct answer is {q['answer']}"
                    )

            st.metric(
                "Your Score",
                f"{score} / {len(questions)}"
            )

            percentage = score / len(questions) * 100

            st.progress(int(percentage))

            if percentage >= 70:
                st.success("Excellent work! 🎉")

            elif percentage >= 40:
                st.info("Good effort! Keep practicing.")

            else:
                st.warning("Practice more and try again.")

    # Footer
    st.divider()
    st.caption(
        "Titanic Dataset Project | Python | PostgreSQL | Streamlit"
    )
