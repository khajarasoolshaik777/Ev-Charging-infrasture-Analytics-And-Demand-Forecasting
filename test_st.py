import sys
import streamlit as st
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())

st.set_page_config(page_title="Test")
