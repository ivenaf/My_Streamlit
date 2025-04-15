import streamlit as st
import random
@st.cache_data
def generate_random_value(x):
    """Generate a random value based on the input."""
    return random.uniform(0, x)
a=generate_random_value(10)
b=generate_random_value(20)
st.write(f"Random value for 10: {a}")
st.write(f"Random value for 20: {b}")