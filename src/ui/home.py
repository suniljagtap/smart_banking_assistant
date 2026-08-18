import json
import requests
import streamlit as st
from streamlit_option_menu import option_menu

# Page Configuration
st.set_page_config(
    page_title="Smart Banking Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# API Configuration
QUERY_API_URL = "http://localhost:8000/api/v1/query/"
QUERY_STREAM_API_URL = "http://localhost:8000/api/v1/query/stream"
UPLOAD_API_URL = "http://localhost:8000/api/v1/admin/upload"

REQUEST_TIMEOUT = 60
UPLOAD_TIMEOUT = 120


# Custom Styling
st.markdown(
    """
    <style>

    /* GLOBAL FONT + APPLICATION */
    html,
    body,
    [class*="css"],
    .stApp {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Roboto,
            Helvetica,
            Arial,
            sans-serif;
    }

    .stApp {
        background: #f6f8fc;
        color: #172b4d;
    }

    .main .block-container {
        max-width: 920px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
    }

    .stAppDeployButton {
        display: none;
    }

    /* STICKY MAIN HEADER */
    .main-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(
            135deg,
            #0f3d68 0%,
            #1265a3 52%,
            #168dcc 100%
        );

        color: white;
        padding: 22px 28px;
        border-radius: 0 0 18px 18px;
        margin: -1.5rem -1rem 28px -1rem;
        box-shadow: 0 8px 28px rgba(15, 61, 104, 0.18);
    }

    .main-header h1 {
        margin: 0;
        color: #ffffff;
        font-size: 28px;
        line-height: 1.25;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-header p {
        margin: 7px 0 0 0;
        color: #e6f4ff;
        font-size: 14px;
        line-height: 1.5;
        font-weight: 400;
        letter-spacing: 0.05px;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: #f8fbff;
        border-right: 1px solid #e5edf5;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .sidebar-title {
        color: #123c69;
        font-size: 19px;
        line-height: 1.3;
        font-weight: 700;
        letter-spacing: -0.2px;
        margin-bottom: 4px;
    }

    .sidebar-subtitle {
        color: #71849a;
        font-size: 13px;
        line-height: 1.5;
        font-weight: 400;
        margin-bottom: 20px;
    }

    /* NAVIGATION */
    div[data-testid="stSidebar"] .nav-link {
        border-radius: 10px !important;
        margin-bottom: 5px;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition:
            background-color 0.2s ease,
            color 0.2s ease;
    }

    div[data-testid="stSidebar"] .nav-link:hover {
        background-color: #e5f1fc !important;
    }

    /* PAGE TITLES */
    .page-title {
        color: #172b4d;
        font-size: 24px;
        line-height: 1.3;
        font-weight: 700;
        letter-spacing: -0.4px;
        margin-bottom: 6px;
    }

    .page-description {
        color: #718096;
        font-size: 14px;
        line-height: 1.6;
        font-weight: 400;
        margin-bottom: 20px;
    }

    /* CONTENT CARD */
    .content-card {
        background: #ffffff;
        padding: 24px 26px;
        border-radius: 16px;
        border: 1px solid #e7edf4;
        box-shadow: 0 4px 16px rgba(31, 73, 104, 0.055);
        margin-bottom: 22px;
    }

    /* CHAT AREA */
    div[data-testid="stChatMessage"] {
        margin-bottom: 12px;
        font-size: 15px;
        line-height: 1.65;
    }

    div[data-testid="stChatMessage"] p {
        font-size: 15px;
        line-height: 1.65;
        color: #243b53;
    }

    /* User / Assistant message spacing */
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-size: 15px;
        line-height: 1.65;
    }

    /* CHAT INPUT */
    div[data-testid="stChatInput"] {
        padding: 6px;
        border-radius: 18px;
        background: transparent;
    }

    div[data-testid="stChatInput"] > div {
        background: #ffffff !important;
        border: 1px solid #d6e1ec !important;
        border-radius: 16px !important;
        box-shadow: 0 5px 20px rgba(31, 73, 104, 0.09);
        padding: 5px 8px;
    }

    div[data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #172b4d !important;
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Roboto,
            Helvetica,
            Arial,
            sans-serif !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
        font-weight: 400 !important;
        padding: 13px 15px !important;
        min-height: 48px !important;
        border: none !important;
        border-radius: 12px !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
        font-size: 14px !important;
        font-weight: 400 !important;
    }

    /* BUTTONS */
    .stButton > button {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Roboto,
            Helvetica,
            Arial,
            sans-serif !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        border-radius: 10px;
        min-height: 42px;
        border: 1px solid #d6e1ec;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #1769aa;
        color: #1769aa;
        box-shadow: 0 4px 12px rgba(23, 105, 170, 0.12);
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #dfe8f1;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stFileUploader"] label {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #334e68 !important;
    }

    /* ALERTS */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        font-size: 14px;
        line-height: 1.55;
    }

    /* SOURCE / CITATIONS */
    .source-text {
        color: #718096;
        font-size: 12px;
        line-height: 1.5;
    }

    div[data-testid="stCaptionContainer"] {
        font-size: 12px;
        line-height: 1.5;
        color: #718096;
    }

    /* DIVIDERS */
    hr {
        border-color: #e5edf5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []


# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 AI-Powered Smart Banking Assistant</h1>
        <p>
            Enter your question below and our AI assistant will 
            help you find the relevant information.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">Banking Assistant</div>
        <div class="sidebar-subtitle">
            Select an option to get started
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = option_menu(
        menu_title=None,
        options=["Chatbot", "File Upload"],
        icons=["chat-dots-fill", "cloud-upload-fill"],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "transparent",
            },
            "icon": {
                "color": "#1769aa",
                "font-size": "17px",
            },
            "nav-link": {
                "font-size": "14px",
                "font-weight": "600",
                "color": "#38556f",
                "padding": "12px 14px",
            },
            "nav-link-selected": {
                "background-color": "#1769aa",
                "color": "white",
            },
        },
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
        type="secondary",
    ):
        st.session_state.messages = []
        st.rerun()


# CHATBOT PAGE
if page == "Chatbot":
    # Display Previous Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User Query
    query = st.chat_input(
        "Ask your banking question...",
        key="query_only",
    )

    if query and query.strip():
        query = query.strip()

        # Display user message immediately
        with st.chat_message("user"):
            st.write(query)

        # Add user message to history
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        # Call Chat API
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "query": query,
                        "chat_history": st.session_state.messages,
                    }

                    response = requests.post(
                        QUERY_API_URL,
                        json=payload,
                        timeout=REQUEST_TIMEOUT,
                    )

                    # HTTP Status Handling
                    if response.status_code != 200:

                        if 400 <= response.status_code < 500:
                            st.error(
                                "The request could not be processed. "
                                "Please check your query and try again."
                            )

                        elif response.status_code >= 500:
                            st.error(
                                "The banking service is temporarily "
                                "unavailable. Please try again later."
                            )

                        else:
                            st.error(
                                "Unable to process the request. " "Please try again."
                            )

                    else:
                        # Parse API JSON
                        try:
                            response_data = response.json()
                        except ValueError:
                            st.error(
                                "The service returned an invalid response. "
                                "Please try again later."
                            )
                            response_data = None

                        if response_data is not None:
                            bot_response = response_data.get("response")

                            if bot_response is None:
                                st.error(
                                    "No response was received from the "
                                    "banking service."
                                )

                            else:
                                # API may return response as: dict or JSON string containing dict
                                if isinstance(bot_response, str):

                                    try:
                                        data = json.loads(bot_response)
                                    except json.JSONDecodeError:
                                        # If backend returns plain text, use it directly as the answer.
                                        data = {
                                            "answer": bot_response,
                                            "citations": [],
                                        }

                                elif isinstance(bot_response, dict):
                                    data = bot_response

                                else:
                                    data = {
                                        "answer": str(bot_response),
                                        "citations": [],
                                    }

                                # Validate answer
                                answer = data.get("answer")

                                if not answer:
                                    st.error(
                                        "The service returned an empty "
                                        "answer. Please try again."
                                    )

                                elif answer == "Not applicable.":
                                    st.warning(
                                        "⚠️ No applicable answer found "
                                        "for this query."
                                    )

                                else:
                                    st.success(answer)

                                    # Citations
                                    citations = data.get("citations", [])

                                    if citations:
                                        st.caption("📚 Sources")
                                        for citation in citations:

                                            if not isinstance(
                                                citation,
                                                dict,
                                            ):
                                                continue

                                            page_number = citation.get(
                                                "page",
                                                "N/A",
                                            )

                                            question = citation.get(
                                                "question",
                                                "Source",
                                            )

                                            st.caption(
                                                f"Page {page_number} — " f"{question}"
                                            )

                                    # Store assistant response
                                    st.session_state.messages.append(
                                        {
                                            "role": "assistant",
                                            "content": answer,
                                        }
                                    )

                # Request Exceptions
                except requests.exceptions.Timeout:
                    st.error("The request timed out. " "Please try again later.")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Unable to connect to the banking service. "
                        "Please make sure the backend service is running."
                    )

                except requests.exceptions.RequestException:
                    st.error(
                        "The banking service is temporarily unavailable. "
                        "Please try again later."
                    )

                except Exception:
                    # Do not expose internal exception details to the end user.
                    st.error(
                        "Something went wrong while processing your "
                        "request. Please try again later."
                    )


# FILE UPLOAD PAGE
elif page == "File Upload":

    st.markdown(
        """
        <div class="content-card">
            <div class="page-title">📄 Knowledgebase Document Upload</div>
            <div class="page-description">
                Upload a PDF documents to make it available
                to the banking assistant.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Only PDF documents are supported.",
    )

    if uploaded_file is not None:

        st.success(f"📄 File selected: **{uploaded_file.name}**")
        st.caption(f"File size: {uploaded_file.size / 1024:.1f} KB")

        if st.button(
            "☁️ Upload File",
            use_container_width=True,
            type="primary",
        ):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }

            with st.spinner("Uploading document..."):
                try:
                    response = requests.post(
                        UPLOAD_API_URL,
                        files=files,
                        timeout=UPLOAD_TIMEOUT,
                    )

                    # Successful Upload
                    if response.status_code in (200, 201):
                        st.success("✅ File successfully uploaded!")

                        # Try to display API response if available
                        try:
                            response_data = response.json()

                            if response_data:
                                with st.expander("Upload Details"):
                                    st.json(response_data)

                        except ValueError:
                            # Successful upload but response isn't JSON.
                            pass

                    # Client Error
                    elif 400 <= response.status_code < 500:
                        error_message = None
                        try:
                            error_data = response.json()
                            if isinstance(error_data, dict):
                                error_message = error_data.get("detail")

                        except ValueError:
                            pass

                        if error_message:
                            st.error(f"Upload failed: {error_message}")
                        else:
                            st.error(
                                "The file could not be uploaded. "
                                "Please check the file and try again."
                            )

                    # Server Error
                    elif response.status_code >= 500:
                        st.error(
                            "The upload service is temporarily "
                            "unavailable. Please try again later."
                        )

                    # Other Status
                    else:
                        st.error("Unable to upload the file. " "Please try again.")

                # Request Exceptions
                except requests.exceptions.Timeout:
                    st.error("File upload timed out. " "Please try again later.")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Unable to connect to the upload service. "
                        "Please make sure the backend service is running."
                    )

                except requests.exceptions.RequestException:
                    st.error(
                        "The upload service is temporarily unavailable. "
                        "Please try again later."
                    )

                except Exception:
                    # Don't expose internal implementation details
                    st.error(
                        "Something went wrong while uploading the file. "
                        "Please try again later."
                    )
