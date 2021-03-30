import pandas as pd
import ipywidgets as widgets
import datetime

def get_date_slider(data, start=None, end=None):
    if start is None:
        start = data.measurements["time"].min().date()
    else:
        start = pd.to_datetime(start).date()
    
    if end is None: 
        end = data.measurements["time"].max().date()
    else:
        end = pd.to_datetime(end).date()

    days_list = pd.date_range(start, end, freq='D')
    days_list_formatted = [(date.strftime(' %d-%m-%y '), date) for date in days_list]

    ld1 = data.events[0]
    start = ld1.start - datetime.timedelta(days=14)
    end = ld1.end + datetime.timedelta(days=14)

    return widgets.SelectionRangeSlider(
        options=days_list_formatted,
        index=(0, len(days_list_formatted) - 1),
        value=(start, end),
        description='Dates',
        orientation='horizontal',
        layout={'width': '500px'}
    )