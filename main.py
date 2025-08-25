import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from io import BytesIO, StringIO
import re

# ------------------------------
# Configuración de la página
# ------------------------------
st.set_page_config(page_title="LLMs con DataFrames", page_icon="📊", layout="wide")
st.title("🤖 LLMs con DataFrames")

# ------------------------------
# Inicialización del modelo
# ------------------------------
model = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=st.secrets["groq"]["API_KEY"],
)

# ------------------------------
# Estado de la sesión
# ------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent" not in st.session_state:
    st.session_state["agent"] = None

def reloadChat():
    st.session_state["messages"] = []
    st.session_state["agent"] = None

# ------------------------------
# Subida de archivo
# ------------------------------
file = st.file_uploader("📂 Elige un archivo CSV", type=["csv"], on_change=reloadChat)

if file is not None:
    df = pd.read_csv(file)
    st.session_state["agent"] = create_pandas_dataframe_agent(
        model, df, allow_dangerous_code=True
    )
    st.success("✅ Archivo cargado correctamente. Ya puedes comenzar a preguntar.")

# ------------------------------
# Renderizado del historial
# ------------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------
# Entrada del usuario
# ------------------------------
if prompt := st.chat_input("Escribe tu pregunta sobre el dataset..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state["agent"] is not None:
        with st.spinner("🔎 Analizando datos..."):
            system_prompt = (
                "Eres un experto en análisis de datos. Responde SIEMPRE en idioma español, "
                "con explicaciones claras y ejemplos cuando sea posible. "
                "Si muestras datos tabulares, usa siempre formato de tabla Markdown."
            )
            query = f"{system_prompt}\n\nPregunta del usuario: {prompt}"
            response = st.session_state["agent"].run(query)

        # Mostrar respuesta del asistente
        with st.chat_message("assistant"):
            # Detectar si la respuesta contiene tabla en markdown
            table_lines = [line for line in response.splitlines() if "|" in line]

            # Eliminar la línea de guiones típica de Markdown (|-----|)
            table_lines = [line for line in table_lines if not re.match(r'^\|\s*-+\s*\|', line.strip())]

            if table_lines:
                try:
                    table_text = "\n".join(table_lines)
                    df_table = pd.read_csv(StringIO(table_text), sep="|", engine="python").dropna(axis=1, how="all")

                    # Limpiar nombres de columnas
                    df_table.columns = [col.strip() for col in df_table.columns]

                    st.write("📊 Tabla detectada en la respuesta:")
                    st.dataframe(df_table)
                except Exception:
                    st.markdown(response)
            else:
                st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})
    else:
        st.error("⚠️ Primero debes subir un archivo CSV para poder hacer preguntas.")

# ------------------------------
# Exportar reporte
# ------------------------------
if st.session_state["messages"]:
    if st.button("📥 Descargar reporte de la conversación"):
        buffer = BytesIO()
        report = "Reporte de Análisis con LLMs\n\n"
        for msg in st.session_state["messages"]:
            role = "👤 Usuario" if msg["role"] == "user" else "🤖 Asistente"
            report += f"{role}:\n{msg['content']}\n\n"

        buffer.write(report.encode("utf-8"))
        buffer.seek(0)

        st.download_button(
            label="📄 Descargar como TXT",
            data=buffer,
            file_name="reporte_llm_dataframe.txt",
            mime="text/plain"
        )
