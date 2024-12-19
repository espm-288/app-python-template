import streamlit as st

st.set_page_config(
    page_title="Streamlit demos",
)

st.sidebar.success("Select a demo above.")

st.title("Exploring LLM Agent Use")

'''
Select any of the demos on the sidebar.  Each illustrates a different way we can incorporate an LLM tool to perform reliable data retrieval (sometimes called retrieval augmented generation, RAG) from specified data resources.  

In this module, you will be adapt one or more of these agents into an interactive application exploring the redlining data we encountered in Module 3 (as seen below). 

'''

import streamlit as st
import leafmap.maplibregl as leafmap
import ibis
from ibis import _
con = ibis.duckdb.connect()


# fixme could create drop-down selection of the 300 cities
city_name = st.text_input("Select a city", "Oakland")

# Extract the specified city 
city = (con
    .read_geo("/vsicurl/https://dsl.richmond.edu/panorama/redlining/static/mappinginequality.gpkg")
    .filter(_.city == city_name, _.residential)
    .execute()
)

# Render the map
m = leafmap.Map(style="positron")
if city_name == "Oakland":
    m.add_cog_layer("https://espm-157-f24.github.io/spatial-carl-amanda-tyler/ndvi.tif", name="ndvi", palette = "greens")
m.add_gdf(city, "fill", paint = {"fill-color": ["get", "fill"], "fill-opacity": 0.8})
m.add_layer_control()
m.to_streamlit()

