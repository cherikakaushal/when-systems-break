import pandas as pd
import streamlit as st


def display_table(frame, percent_columns=None, height=None):
    """Render a responsive themed dataframe with optional percentage formatting."""
    if frame is None or frame.empty:
        st.info("No table data available.")
        return
    output = frame.copy()
    for column in percent_columns or []:
        if column in output.columns:
            output[column] = output[column].map(lambda value: f"{value * 100:.2f}%")
    st.dataframe(output, width="stretch", hide_index=True, height=height)


def csv_download(frame, filename, label="Download CSV"):
    """Render a CSV download button for a dataframe."""
    if frame is None or frame.empty:
        return
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )


def dataframe_to_csv_bytes(frame):
    """Serialize a dataframe to UTF-8 CSV bytes."""
    return frame.to_csv(index=False).encode("utf-8")


def model_table_from_failure(frame):
    if frame.empty:
        return frame
    return frame.round(2)
