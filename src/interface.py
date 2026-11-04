import streamlit as st

class WebUI:
    def __init__(self) -> None:
        st.set_page_config(page_title="Predict Dropout", page_icon="🏫")

        self.col_set, self.col_get = st.columns(2)
        self.result_ui = False

    def build_interface(self):
        st.title("Hello")

        with self.col_set:
            with st.container(border=True):
                age = st.slider("Age :", 0, 99, 18, 1)
                gpa = st.slider("GPA :", 0.00, 4.00, 2.00, 0.01)
                attendance_rate = st.slider("Taux de présence en cours (%) :", 0, 100, 80, 1)
                tuition_up_to_date = st.checkbox("Frais de scolarité a jour.")
                scholarship_holder = st.checkbox("Bénéficie d'une bourse.")
                part_time_job = st.checkbox("Travail à temps partiel.")
                if st.button(label="Lancé la prediction"):
                    pass


        with self.col_get:
            if self.result_ui:
                pass
            else:
                st.info('Entre les données pour faire la prédiction.', icon="ℹ️")




if __name__ == "__main__":
    app = WebUI()
    app.build_interface()
